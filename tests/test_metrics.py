from lab.metrics import aggregate
from lab.metrics import full_blocks
from lab.metrics import median
from lab.metrics import percentile
from lab.metrics import tpot
from lab.settings import load_config


def test_median_even_averages_center():
    assert median([1, 4, 2, 9]) == 3


def test_median_odd_is_middle():
    assert median([3, 1, 2]) == 2


def test_p95_nearest_rank():
    values = [1, 2, 3, 4, 5, 6, 7, 8]
    assert percentile(values, 95) == 8
    assert percentile(values, 50) == 4.5


def test_tpot_excludes_first_token():
    assert abs(tpot(10.0, 10.63, 64) - 0.01) < 1e-12


def test_tpot_needs_two_tokens():
    assert tpot(1.0, 2.0, 1) is None


def test_aggregate_throughput_uses_wall_window():
    requests = [
        {
            "ok": True,
            "t0_s": 1.0,
            "t_done_s": 2.0,
            "ttft_s": 0.2,
            "tpot_s": 0.02,
            "e2e_s": 1.0,
            "prompt_tokens": 100,
            "completion_tokens": 64,
            "chunks": 64,
        },
        {
            "ok": True,
            "t0_s": 1.0,
            "t_done_s": 3.0,
            "ttft_s": 0.4,
            "tpot_s": 0.04,
            "e2e_s": 2.0,
            "prompt_tokens": 100,
            "completion_tokens": 64,
            "chunks": 64,
        },
    ]
    stats = aggregate(requests)
    assert stats["n"] == 2
    assert abs(stats["ttft_p50_s"] - 0.3) < 1e-12
    assert stats["out_tok_s"] == 128 / 2
    assert stats["window_s"] == 2


def test_full_blocks_truncates_tail():
    assert full_blocks(1900, 784) == 1568
    assert full_blocks(422, 784) == 0


def test_prompt_bounds_match_server_caps():
    config = load_config()
    assert config["short_prompt_max"] == config["block_tokens"]
    assert config["long_prompt_min"] == 2 * config["block_tokens"]
    assert 4 * config["long_prompt_max"] <= config["max_num_batched_tokens"]
    assert 8 * config["long_prompt_min"] > config["max_num_batched_tokens"]
