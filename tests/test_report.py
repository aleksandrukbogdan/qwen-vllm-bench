import json

from lab.failures import failure_counts
from lab.failures import failure_rows
from lab.metrics import aggregate
from lab.metrics import full_blocks
from lab.report import dev_report_path
from lab.report import failures_path
from lab.report import parse_main_table
from lab.report import table_matches
from lab.settings import load_config


def _report():
    return json.loads(dev_report_path().read_text(encoding="utf-8"))


def test_latency_replays_from_requests():
    report = _report()
    for name, cell in report["cells"].items():
        stats = aggregate(cell["requests"])
        for key in ("n", "ttft_p50_s", "tpot_p50_s", "out_tok_s"):
            assert abs(stats[key] - cell[key]) < 1e-9, name


def test_cache_mean_is_hit_delta():
    report = _report()
    for name, cell in report["cells"].items():
        assert abs(cell["cached_mean"] - cell["hits_delta"] / cell["n"]) < 1e-9, name
        prompt_sum = sum(row["prompt_tokens"] for row in cell["requests"])
        assert abs(cell["foreign_prompt_tokens"] - (cell["prompt_tokens_delta"] - prompt_sum)) < 1e-6


def test_warm_hit_is_two_pages():
    config = load_config()
    report = _report()
    for name in ("c1_long_hit", "c4_long_hit", "c8_long_hit", "load_c4"):
        cell = report["cells"][name]
        pages = full_blocks(cell["prompt_tokens_mean"], config["block_tokens"])
        assert pages == 2 * config["block_tokens"]
        assert abs(cell["cached_mean"] - pages) < 1e-6
        assert cell["prompt_tokens_mean"] < config["long_prompt_max"]


def test_cold_counters_differ_by_one_prompt():
    report = _report()
    cell = report["cells"]["c8_cold"]
    pages = full_blocks(cell["prompt_tokens_mean"], load_config()["block_tokens"])
    assert cell["hits_delta"] - cell["cached_tokens_delta"] == pages
    assert cell["queries_delta"] - cell["prompt_tokens_delta"] == cell["prompt_tokens_mean"]


def test_readme_table_matches_report():
    readme = dev_report_path().parents[1].joinpath("README.md").read_text(encoding="utf-8")
    report = _report()
    problems = table_matches(readme, report)
    assert not problems, problems
    parsed = parse_main_table(readme)
    assert set(parsed) >= set(report["cells"])


def test_failure_tags_match_report():
    report = _report()
    counts = failure_counts(failure_rows(report))
    assert counts == report["failure_counts"]
    stored = {"rows": 0}
    for line in failures_path().read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        stored["rows"] += 1
        for tag in json.loads(line)["tags"]:
            stored[tag] = stored.get(tag, 0) + 1
    assert stored == report["failure_counts"]
