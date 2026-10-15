"""Maintain and run the Day6-Day10 MARL learning exercises."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Exercise:
    key: str
    path: str
    checks: int
    title: str


EXERCISES = (
    Exercise("ctde", "Day6/11_CTDE_dataflow_test.ipynb", 32, "CTDE data flow"),
    Exercise("mappo", "Day6/12_MAPPO_training_test.ipynb", 56, "One-step cooperation"),
    Exercise("communication", "Day6/13_communication_training_test.ipynb", 41, "Discrete messages"),
    Exercise("qmix-update", "Day7/14_QMIX_single_step_update_test.ipynb", 36, "One QMIX-style update"),
    Exercise("qmix", "Day7/15_QMIX_training_test.ipynb", 23, "Two-round QMIX-style training"),
    Exercise("maddpg", "Day8/16_MADDPG_training_test.ipynb", 24, "Continuous coordination"),
    Exercise("memory", "Day9/17_Recurrent_MAPPO_memory_test.ipynb", 39, "Recurrent PPO memory"),
    Exercise("recurrent-communication", "Day10/18_Recurrent_communication_test.ipynb", 40,
             "Delayed XOR communication"),
)
RESULT_MARKER = "MARL_RUN_JSON="


def load_notebook(path):
    import nbformat

    data = json.loads(path.read_text(encoding="utf-8"))
    # JetBrains adds this display flag outside the notebook output schema.
    # Normalize in memory only; the learner's saved notebook stays untouched.
    for cell in data["cells"]:
        for output in cell.get("outputs", []):
            output.pop("jetTransient", None)
    notebook = nbformat.reads(json.dumps(data), as_version=4)
    nbformat.validate(notebook)
    return notebook


def notebook_code(notebook):
    return "\n\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")


def code_ast(code):
    return ast.dump(ast.parse(code), include_attributes=False)


def export_source(notebook):
    sections = ["# Generated from the adjacent notebook by scripts/marl.py export.\n"]
    for cell in notebook.cells:
        if cell.cell_type == "code":
            sections.append("# %%\n" + cell.source.rstrip() + "\n")
        elif cell.cell_type == "markdown":
            comments = "\n".join("# " + line if line else "#" for line in cell.source.splitlines())
            sections.append("# %% [markdown]\n" + comments + "\n")
    return "\n".join(sections)


def selected_exercises(selection):
    return EXERCISES if selection == "all" else tuple(e for e in EXERCISES if e.key == selection)


def check_links():
    paths = [ROOT / "README.md"]
    paths.extend((ROOT / "docs").rglob("*.md"))
    count = 0
    for path in paths:
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"(?ms)^```[^\n]*\n.*?^```[^\n]*(?:\n|$)", "", text)
        text = re.sub(r"`[^`\n]+`", "", text)
        for target in re.findall(r"!?\[[^\]]*\]\(([^)\n]+)\)", text):
            target = target.strip().strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            destination = (path.parent / unquote(parsed.path)).resolve()
            if not destination.is_file():
                raise ValueError(f"Broken link in {path.relative_to(ROOT)}: {target}")
            count += 1
    return count


def validate_exercises():
    for exercise in EXERCISES:
        path = ROOT / exercise.path
        notebook = load_notebook(path)
        script = path.with_suffix(".py")
        if code_ast(notebook_code(notebook)) != code_ast(script.read_text(encoding="utf-8")):
            raise ValueError(f"Notebook/script code differs: {exercise.key}; run export intentionally.")
        print(f"PASS {exercise.key}: notebook schema, Python syntax and matching script", flush=True)
    print(f"PASS {check_links()} local documentation links", flush=True)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(namespace, expected):
    summary = namespace.get("summary", namespace.get("result"))
    if not isinstance(summary, dict):
        raise ValueError("Exercise did not produce a summary/result dictionary.")
    actual = summary.get("checks_passed", summary.get("check_count"))
    if actual is None and isinstance(summary.get("checks"), list):
        actual = len(summary["checks"])
    if actual != expected:
        raise ValueError(f"Expected {expected} embedded checks, got {actual}.")
    total = summary.get("checks_total")
    if total is not None and total != actual:
        raise ValueError("Some embedded checks did not pass.")
    return summary


SCRIPT_BOOTSTRAP = r'''
import json
from pathlib import Path
import runpy
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

output = Path(sys.argv[2])
figure_count = 0
def save_figures(*args, **kwargs):
    global figure_count
    for number in plt.get_fignums():
        figure_count += 1
        plt.figure(number).savefig(output / f"figure-{figure_count:02d}.png", dpi=150)
    plt.close("all")

plt.show = save_figures
namespace = runpy.run_path(sys.argv[1], run_name="__main__")
save_figures()
result = namespace.get("summary", namespace.get("result"))
print("\nMARL_RUN_JSON=" + json.dumps(result, ensure_ascii=True, allow_nan=False))
'''


def execute_script(exercise, output, timeout):
    environment = os.environ.copy()
    environment.update(MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    if os.name == "nt":
        environment["MKL_THREADING_LAYER"] = "SEQUENTIAL"
    with (output / "stdout.txt").open("w", encoding="utf-8") as stdout, (
        output / "stderr.txt"
    ).open("w", encoding="utf-8") as stderr:
        process = subprocess.run(
            [sys.executable, "-u", "-c", SCRIPT_BOOTSTRAP,
             str((ROOT / exercise.path).with_suffix(".py")), str(output)],
            cwd=ROOT, env=environment, stdout=stdout, stderr=stderr,
            timeout=timeout,
        )
    if process.returncode:
        raise RuntimeError(f"Script exited {process.returncode}; see {output / 'stderr.txt'}")
    summaries = [
        json.loads(line[len(RESULT_MARKER):])
        for line in (output / "stdout.txt").read_text(encoding="utf-8").splitlines()
        if line.startswith(RESULT_MARKER)
    ]
    if len(summaries) != 1:
        raise ValueError("Script did not emit exactly one maintenance summary.")
    return summarize({"summary": summaries[0]}, exercise.checks)


def execute_notebook(exercise, output, timeout):
    if sys.platform == "win32":
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    import nbformat
    from nbclient import NotebookClient

    notebook = load_notebook(ROOT / exercise.path)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None
    notebook.cells.insert(0, nbformat.v4.new_code_cell(
        "import os\n"
        "if os.name == 'nt':\n"
        "    os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'\n"
        "import matplotlib\n"
        "matplotlib.use('module://matplotlib_inline.backend_inline')"
    ))
    notebook.cells.append(nbformat.v4.new_code_cell(
        "import json\n"
        f"print('{RESULT_MARKER}' + json.dumps("
        "globals().get('summary', globals().get('result')), ensure_ascii=True, allow_nan=False))"
    ))
    try:
        NotebookClient(
            notebook, timeout=timeout, kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
            record_timing=False,
        ).execute()
    finally:
        nbformat.write(notebook, output / "executed.ipynb")
    streams = "\n".join(
        item.text for cell in notebook.cells for item in cell.get("outputs", [])
        if item.output_type == "stream"
    )
    (output / "stdout.txt").write_text(streams, encoding="utf-8")
    summaries = [
        json.loads(line[len(RESULT_MARKER):])
        for line in streams.splitlines() if line.startswith(RESULT_MARKER)
    ]
    if len(summaries) != 1:
        raise ValueError("Notebook did not emit exactly one maintenance summary.")
    return summarize({"summary": summaries[0]}, exercise.checks)


def run_exercises(args):
    destination = Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "report.json"
    if report_path.exists():
        raise ValueError("Choose a fresh output directory; an earlier report already exists.")
    results = []
    for exercise in selected_exercises(args.exercise):
        source = ROOT / exercise.path
        script = source.with_suffix(".py")
        before = (digest(source), digest(script))
        output = destination / exercise.key
        output.mkdir(exist_ok=True)
        start = time.monotonic()
        row = {"exercise": exercise.key, "engine": args.engine,
               "source": exercise.path, "source_sha256": before[0],
               "script_sha256": before[1], "expected_checks": exercise.checks}
        print(f"RUN {exercise.key} ({args.engine})", flush=True)
        try:
            if code_ast(notebook_code(load_notebook(source))) != code_ast(
                script.read_text(encoding="utf-8")
            ):
                raise ValueError("Notebook/script code differs; refusing to run stale code.")
            execute = execute_notebook if args.engine == "notebook" else execute_script
            row["summary"] = execute(exercise, output, args.timeout)
            row["status"] = "passed"
        except Exception as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
        if before != (digest(source), digest(script)):
            row.update(status="failed", error="Source changed during execution.")
        row["duration_seconds"] = round(time.monotonic() - start, 3)
        results.append(row)
        report_path.write_text(json.dumps({
            "executed_at": datetime.now().astimezone().isoformat(),
            "python": sys.version.split()[0], "results": results,
        }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"{row['status'].upper()} {exercise.key}: {row['duration_seconds']}s", flush=True)
        if row["status"] == "failed":
            raise RuntimeError(row["error"])
    print(f"PASS {len(results)} exercises, "
          f"{sum(row['expected_checks'] for row in results)} embedded checks; {report_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List the eight maintained exercises.")
    subparsers.add_parser("check", help="Validate notebook/script consistency and local links.")
    subparsers.add_parser("export", help="Create missing scripts; preserve matching existing scripts.")
    runner = subparsers.add_parser("run", help="Run complete default exercises in isolated processes/kernels.")
    runner.add_argument("exercise", choices=["all"] + [e.key for e in EXERCISES], default="all", nargs="?")
    runner.add_argument("--engine", choices=("script", "notebook"), default="script")
    runner.add_argument("--timeout", type=int, default=900)
    runner.add_argument("--output", default=str(ROOT / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")))
    args = parser.parse_args()
    if not __debug__:
        parser.error("Do not use python -O: exercises rely on assertions.")
    if args.command == "list":
        for exercise in EXERCISES:
            print(f"{exercise.key:25s} {exercise.path} ({exercise.checks} checks)")
    elif args.command == "check":
        validate_exercises()
    elif args.command == "export":
        for exercise in EXERCISES:
            notebook = load_notebook(ROOT / exercise.path)
            target = (ROOT / exercise.path).with_suffix(".py")
            if target.exists():
                if code_ast(target.read_text(encoding="utf-8")) != code_ast(notebook_code(notebook)):
                    raise ValueError(f"{target.name} has different code; reconcile your edits first.")
                print(f"KEEP {target.name}: matching code, existing comments preserved")
            else:
                target.write_text(export_source(notebook), encoding="utf-8")
                print(f"EXPORT {target.name}")
    else:
        run_exercises(args)


if __name__ == "__main__":
    main()
