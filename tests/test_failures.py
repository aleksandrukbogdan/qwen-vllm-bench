from lab.failures import cell_tags
from lab.failures import request_tags


def _cell(**overrides):
    cell = {
        "name": "c1_long_hit",
        "prefix": "hit",
        "concurrency": 1,
        "cached_mean": 1568,
        "prompt_tokens_mean": 1800,
        "waiting_max": 0,
        "foreign_prompt_tokens": 0,
        "ttft_p50_s": 0.2,
        "requests": [],
    }
    cell.update(overrides)
    return cell


def test_long_hit_is_a_partial_page():
    assert "partial_page" in cell_tags(_cell())


def test_short_hit_below_block():
    tags = cell_tags(_cell(name="c1_short_hit", prefix="short", cached_mean=0, prompt_tokens_mean=400))
    assert tags == ["below_block"]


def test_queue_is_over_cap():
    tags = cell_tags(
        _cell(
            name="queue_c12",
            prefix="short",
            concurrency=12,
            cached_mean=0,
            prompt_tokens_mean=400,
            waiting_max=2,
        )
    )
    assert "over_cap" in tags
    assert "queued" in tags
    assert "below_block" in tags


def test_wide_prefill_is_chunked():
    tags = cell_tags(_cell(name="c8_long_miss", prefix="miss", concurrency=8, cached_mean=0, prompt_tokens_mean=1800))
    assert "chunked" in tags
    assert "cold_prefix" in tags


def test_slow_request_uses_median_and_floor():
    cell = _cell(
        requests=[
            {
                "ok": True,
                "i": 0,
                "ttft_s": 0.2,
                "prompt_tokens": 1800,
                "completion_tokens": 64,
            },
            {
                "ok": True,
                "i": 1,
                "ttft_s": 1.5,
                "prompt_tokens": 1800,
                "completion_tokens": 64,
            },
        ]
    )
    rows = request_tags(cell)
    assert len(rows) == 1
    assert rows[0]["i"] == 1
    assert rows[0]["tags"] == ["slow"]
