"""Write a job summary from a pytest junit file and enforce a suite time budget.

Usage: python ci_summary.py junit.xml [--title TITLE] [--budget-seconds N]

Appends markdown to $GITHUB_STEP_SUMMARY (stdout when unset). Exits non-zero when the
suite's wall time exceeds the budget, or when any test was skipped (a skip hides a
missing dependency). The wall time is the junit testsuite time, which pytest measures
over the whole session (collection included) and which stays correct under
pytest-xdist; the summed per-test time is reported alongside it.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET


def main():
    """Summarize collected tests and check the requested budget and skip policy."""
    parser = argparse.ArgumentParser()
    parser.add_argument("junit")
    parser.add_argument("--title", default="Tests")
    parser.add_argument("--budget-seconds", type=float, default=0.0)
    parser.add_argument("--allow-skips", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.junit):
        print(f"{args.junit} not found: the test step did not run")
        return 1
    root = ET.parse(args.junit).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    wall = sum(float(suite.get("time", 0.0)) for suite in suites)
    cases = []
    skipped = []
    for case in root.iter("testcase"):
        name = f"{case.get('classname', '')}::{case.get('name', '')}"
        seconds = float(case.get("time", 0.0))
        cases.append((seconds, name))
        # Pytest records expected failures as skipped; pytest enforces strict xfails.
        skip = case.find("skipped")
        if skip is not None and skip.get("type") != "pytest.xfail":
            skipped.append(name)
    total = sum(s for s, _ in cases)
    cases.sort(reverse=True)

    lines = [
        f"### {args.title}",
        "",
        f"- tests: {len(cases)}, skipped: {len(skipped)}",
        f"- wall time: {wall:.1f} s"
        + (f" (budget {args.budget_seconds:.0f} s)" if args.budget_seconds else ""),
        f"- summed test time: {total:.1f} s",
        "",
        "| slowest tests | seconds |",
        "|---|---|",
    ]
    lines += [f"| `{name}` | {seconds:.2f} |" for seconds, name in cases[:10]]
    if skipped:
        policy = "allowed" if args.allow_skips else "treated as a failure"
        lines += ["", f"Skipped ({policy}):"]
        lines += [f"- `{name}`" for name in skipped[:20]]

    text = "\n".join(lines) + "\n"
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if target:
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(text)
    else:
        print(text)

    failed = False
    if args.budget_seconds and wall > args.budget_seconds:
        print(f"wall time {wall:.1f} s exceeds budget {args.budget_seconds:.0f} s")
        failed = True
    if skipped and not args.allow_skips:
        print(f"{len(skipped)} tests were skipped")
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
