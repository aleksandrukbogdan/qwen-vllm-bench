import math


def median(values):
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        raise ValueError("пустая выборка")
    if count % 2 == 0:
        mid = count // 2
        return (ordered[mid - 1] + ordered[mid]) / 2
    return ordered[count // 2]


def percentile(values, percent):
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        raise ValueError("пустая выборка")
    if percent == 50:
        return median(ordered)
    rank = math.ceil(percent / 100 * count) - 1
    rank = min(max(rank, 0), count - 1)
    return ordered[rank]


def tpot(t_first, t_last, completion_tokens):
    if t_first is None or t_last is None or completion_tokens < 2:
        return None
    return (t_last - t_first) / (completion_tokens - 1)


def aggregate(requests):
    ok = [row for row in requests if row["ok"]]
    if not ok:
        raise ValueError("в ячейке нет успешных запросов")
    window = max(row["t_done_s"] for row in ok) - min(row["t0_s"] for row in ok)
    tokens = sum(row["completion_tokens"] for row in ok)
    if window <= 0:
        raise ValueError("окно ячейки нулевое")
    ttfts = [row["ttft_s"] for row in ok]
    tpots = [row["tpot_s"] for row in ok]
    return {
        "n": len(ok),
        "ttft_p50_s": median(ttfts),
        "ttft_p95_s": percentile(ttfts, 95),
        "ttft_min_s": min(ttfts),
        "ttft_max_s": max(ttfts),
        "tpot_p50_s": median(tpots),
        "tpot_p95_s": percentile(tpots, 95),
        "e2e_p50_s": median([row["e2e_s"] for row in ok]),
        "out_tok_s": tokens / window,
        "prompt_tokens_mean": sum(row["prompt_tokens"] for row in ok) / len(ok),
        "completion_tokens_mean": tokens / len(ok),
        "chunks_mean": sum(row["chunks"] for row in ok) / len(ok),
        "window_s": window,
    }


def full_blocks(tokens, block):
    return (int(tokens) // block) * block
