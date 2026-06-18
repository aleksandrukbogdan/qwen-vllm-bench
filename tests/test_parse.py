from lab.client import empty_state
from lab.client import note_event
from lab.client import parse_sse_line
from lab.counters import parse_prometheus
from lab.metrics import tpot


def test_sse_skips_role_and_keeps_usage():
    assert parse_sse_line("") == ("skip", None)
    assert parse_sse_line("data: [DONE]") == ("done", None)
    kind, event = parse_sse_line('data: {"choices":[{"delta":{"content":"А"}}]}')
    assert kind == "event"
    state = empty_state()
    note_event(state, {"choices": [{"delta": {"role": "assistant"}, "index": 0}]}, 1.0)
    assert state["t_first"] is None
    note_event(state, event, 1.2)
    note_event(state, {"choices": [{"delta": {"content": "б"}, "finish_reason": "length"}]}, 2.46)
    note_event(
        state,
        {"usage": {"prompt_tokens": 10, "completion_tokens": 64}, "choices": []},
        2.5,
    )
    assert state["chunks"] == 2
    assert state["t_first"] == 1.2
    assert state["finish_reason"] == "length"
    assert state["usage"]["prompt_tokens"] == 10
    assert tpot(state["t_first"], state["t_last"], 64) == (2.46 - 1.2) / 63


def test_prometheus_picks_counters():
    text = """
# HELP vllm:prefix_cache_hits_total cached
vllm:prefix_cache_hits_total{engine="0",model_name="Qwen3-VL"} 1568
vllm:prefix_cache_queries_total{engine="0",model_name="Qwen3-VL"} 1900
vllm:prompt_tokens_total{engine="0",model_name="Qwen3-VL"} 1900
vllm:prompt_tokens_cached_total{engine="0",model_name="Qwen3-VL"} 1568
vllm:prompt_tokens_by_source_total{engine="0",model_name="Qwen3-VL",source="local"} 9
vllm:num_requests_running{engine="0",model_name="Qwen3-VL"} 2
vllm:num_requests_waiting{engine="0",model_name="Qwen3-VL"} 0
vllm:kv_cache_usage_perc{engine="0",model_name="Qwen3-VL"} 0.01
"""
    parsed = parse_prometheus(text)
    assert parsed["vllm:prefix_cache_hits_total"] == 1568
    assert parsed["vllm:num_requests_running"] == 2
    assert parsed["vllm:prompt_tokens_total"] == 1900
