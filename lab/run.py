import json
import platform
import socket
import subprocess
import time
from datetime import datetime, timezone

from lab.client import StreamClient
from lab.client import discover_model
from lab.client import tokenize
from lab.counters import subtract
from lab.failures import failure_counts
from lab.failures import failure_rows
from lab.metrics import aggregate
from lab.prompts import corpus_lines
from lab.prompts import sha256_text
from lab.prompts import system_text
from lab.prompts import user_text
from lab.report import markdown_table
from lab.report import write_json
from lab.report import cell_plan
from lab.report import dev_report_path
from lab.report import failures_path
from lab.settings import load_config
from lab.settings import root as project_root
from lab.workload import run_cell


def api_root(base_url):
    text = base_url.rstrip("/")
    if text.endswith("/v1"):
        return text[:-3]
    return text


def gpu_snapshot():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        index, used, total, util = [part.strip() for part in line.split(",")]
        rows.append(
            {
                "index": int(index),
                "memory_used_mib": int(used),
                "memory_total_mib": int(total),
                "utilization_gpu": int(util),
            }
        )
    return rows


def compact_warmup(row):
    if row is None:
        return None
    return {
        "ttft_s": row["ttft_s"],
        "tpot_s": row["tpot_s"],
        "e2e_s": row["e2e_s"],
        "prompt_tokens": row["prompt_tokens"],
        "completion_tokens": row["completion_tokens"],
    }


def check_bounds(name, prompt_tokens, config):
    if name == "c1_short_hit" and prompt_tokens >= config["short_prompt_max"]:
        raise SystemExit(
            f"короткий промпт {prompt_tokens} токенов не ниже страницы {config['short_prompt_max']}"
        )
    if name == "c1_long_hit":
        low = config["long_prompt_min"]
        high = config["long_prompt_max"]
        if not low <= prompt_tokens <= high:
            raise SystemExit(f"длинный промпт {prompt_tokens} токенов вне [{low}, {high}]")


def run_dev(base_url=None):
    config = load_config()
    base = (base_url or config["base_url"]).rstrip("/")
    model_row = discover_model(base)
    if model_row.get("id") != config["model"]:
        raise SystemExit(f"ожидалась модель {config['model']}, сервер отдал {model_row.get('id')}")
    weight_root = model_row.get("root") or ""
    if "FP8" not in weight_root:
        raise SystemExit(f"ожидались веса FP8, root={weight_root}")
    lines = corpus_lines(config["long_lines"])
    frozen = [line for line in (project_root() / "data" / "prefix.txt").read_text(encoding="utf-8").splitlines() if line]
    if frozen != lines:
        raise SystemExit("data/prefix.txt не совпадает с генератором")
    hit_text = system_text("hit", 0, lines)
    client = StreamClient(
        base,
        config["model"],
        config["timeout_s"],
        config["temperature"],
        config["seed"],
        config["max_tokens"],
    )
    metrics_url = api_root(base) + "/metrics"
    raw_tokens = {
        "short": tokenize(base, config["model"], system_text("short", 0, lines)),
        "hit": tokenize(base, config["model"], hit_text),
        "cold": tokenize(base, config["model"], system_text("cold", 0, lines)),
        "user": tokenize(base, config["model"], user_text(0)),
    }
    started = datetime.now(timezone.utc).isoformat()
    running_at_start = None
    try:
        from lab.counters import scrape

        running_at_start = scrape(metrics_url)["vllm:num_requests_running"]
    except Exception as exc:
        running_at_start = f"error: {exc}"
    cells = {}
    print(f"model {model_row.get('id')} root {weight_root} raw_tokens {raw_tokens}", flush=True)
    for spec in cell_plan():
        kind = spec.get("kind") or spec["prefix"]
        index_base = spec.get("index_base", 0)
        warmup_box = {}

        def call(job, warmup, kind=kind, index_base=index_base, warmup_box=warmup_box, n_req=spec["requests"]):
            index = index_base + n_req if warmup else index_base + job["i"]
            row = client.stream(system_text(kind, index, lines), user_text(index))
            row["i"] = job["i"]
            row["prompt_index"] = index
            if warmup:
                warmup_box["row"] = row
            return row

        print(f"cell {spec['name']} concurrency {spec['concurrency']} n {spec['requests']}", flush=True)
        t0 = time.perf_counter()
        rows, before, after, sampler = run_cell(spec, call, metrics_url)
        failed = [row for row in rows if not row["ok"]]
        if failed:
            raise SystemExit(f"{spec['name']} ошибка запроса {failed[0]['i']}: {failed[0].get('error')}")
        if any(row["tpot_s"] is None for row in rows):
            raise SystemExit(f"{spec['name']}: нечем считать TPOT")
        warmup = warmup_box.get("row")
        if spec["warmup"]:
            if warmup is None or not warmup["ok"]:
                raise SystemExit(f"{spec['name']}: прогрев не удался")
            check_bounds(spec["name"], warmup["prompt_tokens"], config)
        stats = aggregate(rows)
        delta = subtract(after, before)
        prompt_sum = sum(row["prompt_tokens"] for row in rows)
        cached_mean = delta["vllm:prefix_cache_hits_total"] / stats["n"]
        cell = {
            "name": spec["name"],
            "concurrency": spec["concurrency"],
            "prefix": spec["prefix"],
            "kind": kind,
            "burst": spec["burst"],
            "warmup": compact_warmup(warmup),
            **stats,
            "cached_mean": cached_mean,
            "hits_delta": delta["vllm:prefix_cache_hits_total"],
            "queries_delta": delta["vllm:prefix_cache_queries_total"],
            "prompt_tokens_delta": delta["vllm:prompt_tokens_total"],
            "cached_tokens_delta": delta["vllm:prompt_tokens_cached_total"],
            "foreign_prompt_tokens": delta["vllm:prompt_tokens_total"] - prompt_sum,
            "running_max": sampler.running_max,
            "waiting_max": sampler.waiting_max,
            "kv_max": sampler.kv_max,
            "gauge_samples": sampler.samples,
            "requests": rows,
        }
        cells[spec["name"]] = cell
        print(
            f"  ttft_p50 {stats['ttft_p50_s']:.4f} tpot_p50 {stats['tpot_p50_s']:.4f} "
            f"out_tok_s {stats['out_tok_s']:.1f} cached_mean {cached_mean:.1f} "
            f"foreign {cell['foreign_prompt_tokens']:.0f} wait {sampler.waiting_max:.0f} "
            f"wall {time.perf_counter() - t0:.1f}s",
            flush=True,
        )
    report = {
        "seed": config["seed"],
        "base_url": base,
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
        "ignore_eos": True,
        "model": {
            "id": model_row.get("id"),
            "root": weight_root,
            "max_model_len": model_row.get("max_model_len"),
        },
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "started_at": started,
        "prefix_sha256": sha256_text(hit_text),
        "raw_tokens": raw_tokens,
        "logged": config["logged"],
        "gpu_before": gpu_snapshot(),
        "running_at_start": running_at_start,
        "cells": cells,
    }
    report["gpu_after"] = gpu_snapshot()
    rows = failure_rows(report)
    report["failure_counts"] = failure_counts(rows)
    write_json(dev_report_path(), report)
    failures_path().parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    failures_path().write_text(payload + ("\n" if payload else ""), encoding="utf-8")
    print(markdown_table(report), flush=True)
    return report
