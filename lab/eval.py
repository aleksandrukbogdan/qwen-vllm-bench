import argparse
import json
import sys
from pathlib import Path

from lab.report import markdown_table
from lab.report import table_matches


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(prog="python -m lab.eval")
    parser.add_argument("--split", choices=("dev", "check"), default="dev")
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args(argv)
    if args.split == "check":
        root = Path(__file__).resolve().parents[1]
        report = json.loads((root / "reports" / "dev.json").read_text(encoding="utf-8"))
        readme = (root / "README.md").read_text(encoding="utf-8")
        problems = table_matches(readme, report)
        print(markdown_table(report))
        if problems:
            for problem in problems:
                print(problem)
            raise SystemExit(1)
        print("table ok")
        return
    from lab.run import run_dev

    run_dev(base_url=args.base_url)


if __name__ == "__main__":
    main()
