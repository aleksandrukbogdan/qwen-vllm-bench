import json
from pathlib import Path

from lab.settings import load_config
from lab.settings import root


CELL_ORDER = (
    "c1_short_hit",
    "c1_long_miss",
    "c1_long_hit",
    "c4_long_miss",
    "c4_long_hit",
    "c8_long_miss",
    "c8_long_hit",
    "c8_cold",
    "load_c4",
    "queue_c12",
)

METRIC_COLUMNS = (
    ("ttft_p50_s", 4),
    ("tpot_p50_s", 4),
    ("out_tok_s", 1),
    ("cached_mean", 1),
)


def cell_plan():
    config = load_config()
    common = config["requests_per_cell"]
    return (
        {"name": "c1_short_hit", "concurrency": 1, "prefix": "short", "requests": common, "burst": False, "warmup": True},
        {"name": "c8_cold", "concurrency": 8, "prefix": "cold", "requests": common, "burst": True, "warmup": False},
        {"name": "c1_long_miss", "concurrency": 1, "prefix": "miss", "requests": common, "burst": False, "warmup": True, "index_base": 500},
        {"name": "c4_long_miss", "concurrency": 4, "prefix": "miss", "requests": common, "burst": False, "warmup": True, "index_base": 600},
        {"name": "c8_long_miss", "concurrency": 8, "prefix": "miss", "requests": common, "burst": False, "warmup": True, "index_base": 700},
        {"name": "c1_long_hit", "concurrency": 1, "prefix": "hit", "requests": common, "burst": False, "warmup": True},
        {"name": "c4_long_hit", "concurrency": 4, "prefix": "hit", "requests": common, "burst": False, "warmup": True},
        {"name": "c8_long_hit", "concurrency": 8, "prefix": "hit", "requests": common, "burst": False, "warmup": True},
        {"name": "load_c4", "concurrency": 4, "prefix": "hit", "requests": config["load_requests"], "burst": False, "warmup": True},
        {"name": "queue_c12", "concurrency": 12, "prefix": "short", "kind": "queue", "requests": 12, "burst": True, "warmup": False, "index_base": 800},
    )


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def dev_report_path():
    return root() / "reports" / "dev.json"


def failures_path():
    return root() / "reports" / "failures.jsonl"


def parse_main_table(text):
    lines = text.splitlines()
    header_i = None
    for index, line in enumerate(lines):
        if line.strip().startswith("|") and "ttft_p50_s" in line and "cached_mean" in line:
            header_i = index
            break
    if header_i is None:
        raise AssertionError("в README нет таблицы метрик")
    rows = {}
    for line in lines[header_i + 2 :]:
        if not line.strip().startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 7:
            continue
        rows[cells[0]] = {
            "concurrency": int(cells[1]),
            "prefix": cells[2],
            "ttft_p50_s": float(cells[3]),
            "tpot_p50_s": float(cells[4]),
            "out_tok_s": float(cells[5]),
            "cached_mean": float(cells[6]),
        }
    return rows


def table_matches(readme, report):
    parsed = parse_main_table(readme)
    problems = []
    plan = {item["name"]: item for item in cell_plan()}
    for name in CELL_ORDER:
        if name not in report["cells"]:
            continue
        if name not in parsed:
            problems.append(f"нет строки {name}")
            continue
        block = report["cells"][name]
        got = parsed[name]
        if got["concurrency"] != plan[name]["concurrency"]:
            problems.append(f"{name} concurrency: в README {got['concurrency']}")
        if got["prefix"] != plan[name]["prefix"]:
            problems.append(f"{name} prefix: в README {got['prefix']}")
        for key, digits in METRIC_COLUMNS:
            expected = round(block[key], digits)
            if round(got[key], digits) != expected:
                problems.append(f"{name} {key}: в README {got[key]}, в отчёте {expected}")
    return problems


def markdown_table(report):
    header = "| experiment | concurrency | prefix | ttft_p50_s | tpot_p50_s | out_tok_s | cached_mean |"
    sep = "| --- | --- | --- | --- | --- | --- | --- |"
    lines = [header, sep]
    plan = {item["name"]: item for item in cell_plan()}
    for name in CELL_ORDER:
        block = report["cells"].get(name)
        if block is None:
            continue
        spec = plan[name]
        cells = [name, str(spec["concurrency"]), spec["prefix"]]
        for key, digits in METRIC_COLUMNS:
            cells.append(f"{block[key]:.{digits}f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
