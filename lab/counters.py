import urllib.request


COUNTER_NAMES = (
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prompt_tokens_total",
    "vllm:prompt_tokens_cached_total",
)

GAUGE_NAMES = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
)


def parse_prometheus(text):
    found = {}
    names = COUNTER_NAMES + GAUGE_NAMES
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        for name in names:
            prefix = name + "{"
            if line.startswith(prefix):
                value = float(line.rsplit(" ", 1)[1])
                found[name] = value
                break
    missing = [name for name in names if name not in found]
    if missing:
        raise RuntimeError("в /metrics нет " + ", ".join(missing))
    return found


def scrape(metrics_url, timeout=30):
    request = urllib.request.Request(metrics_url)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        text = response.read().decode("utf-8", errors="replace")
    return parse_prometheus(text)


def subtract(after, before):
    return {name: after[name] - before[name] for name in COUNTER_NAMES}
