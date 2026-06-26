from lab.metrics import full_blocks
from lab.settings import load_config


def cell_tags(cell):
    config = load_config()
    tags = []
    cached = cell["cached_mean"]
    prompt = cell["prompt_tokens_mean"]
    block = config["block_tokens"]
    if cell["prefix"] == "short" and cached < 1:
        tags.append("below_block")
    if cell["prefix"] == "hit" and cached > 0 and cached + 32 < prompt:
        pages = full_blocks(prompt, block)
        if abs(cached - pages) <= block * 0.25:
            tags.append("partial_page")
        else:
            tags.append("cache_off_page")
    if cell["prefix"] == "miss" and cached < 1:
        tags.append("cold_prefix")
    if cell["prefix"] == "miss" and cached >= 1:
        tags.append("unexpected_hit")
    if cell["prefix"] == "cold" and cached < block:
        tags.append("stampede")
    if cell["waiting_max"] >= 1:
        tags.append("queued")
    if cell["concurrency"] > config["max_num_seqs"]:
        tags.append("over_cap")
    if cell["foreign_prompt_tokens"] > config["foreign_prompt_tokens"]:
        tags.append("foreign")
    if cell["concurrency"] * prompt > config["max_num_batched_tokens"]:
        tags.append("chunked")
    return tags


def request_tags(cell):
    config = load_config()
    median = cell["ttft_p50_s"]
    limit = max(config["slow_ttft_floor_s"], config["slow_ttft_factor"] * median)
    rows = []
    for req in cell["requests"]:
        tags = []
        if not req["ok"]:
            tags.append("error")
        elif req["completion_tokens"] != config["max_tokens"]:
            tags.append("short_completion")
        if req["ok"] and req["ttft_s"] > limit:
            tags.append("slow")
        if tags:
            rows.append(
                {
                    "kind": "request",
                    "name": cell["name"],
                    "i": req["i"],
                    "tags": tags,
                    "ttft_s": req["ttft_s"],
                    "prompt_tokens": req["prompt_tokens"],
                    "completion_tokens": req["completion_tokens"],
                }
            )
    return rows


def failure_rows(report):
    rows = []
    for name, cell in report["cells"].items():
        tags = cell_tags(cell)
        if tags:
            rows.append(
                {
                    "kind": "cell",
                    "name": name,
                    "tags": tags,
                    "ttft_p50_s": cell["ttft_p50_s"],
                    "tpot_p50_s": cell["tpot_p50_s"],
                    "cached_mean": cell["cached_mean"],
                    "prompt_tokens_mean": cell["prompt_tokens_mean"],
                    "waiting_max": cell["waiting_max"],
                    "foreign_prompt_tokens": cell["foreign_prompt_tokens"],
                }
            )
        rows.extend(request_tags(cell))
    return rows


def failure_counts(rows):
    counts = {"rows": len(rows)}
    for row in rows:
        for tag in row["tags"]:
            counts[tag] = counts.get(tag, 0) + 1
    return counts
