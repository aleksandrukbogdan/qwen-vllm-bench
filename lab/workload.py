import queue
import threading
import time

from lab.counters import scrape


def safe_call(fn, job):
    try:
        row = fn(job)
        row["ok"] = True
        return row
    except Exception as exc:
        now = time.perf_counter()
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "i": job["i"],
            "t0": now,
            "t_done": now,
            "ttft_s": None,
            "tpot_s": None,
            "e2e_s": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "chunks": 0,
            "finish_reason": None,
            "text": "",
        }


def closed_loop(jobs, concurrency, fn):
    pending = queue.Queue()
    for job in jobs:
        pending.put(job)
    results = [None] * len(jobs)

    def worker():
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            results[job["i"]] = safe_call(fn, job)

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def burst(jobs, fn):
    barrier = threading.Barrier(len(jobs))
    results = [None] * len(jobs)

    def worker(job):
        barrier.wait()
        results[job["i"]] = safe_call(fn, job)

    threads = [threading.Thread(target=worker, args=(job,)) for job in jobs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


class GaugeSampler:
    def __init__(self, metrics_url, interval=0.3):
        self.metrics_url = metrics_url
        self.interval = interval
        self.stop = threading.Event()
        self.running_max = 0.0
        self.waiting_max = 0.0
        self.kv_max = 0.0
        self.samples = 0
        self.errors = 0
        self.thread = threading.Thread(target=self._loop)

    def _loop(self):
        while not self.stop.is_set():
            try:
                sample = scrape(self.metrics_url)
                self.running_max = max(self.running_max, sample["vllm:num_requests_running"])
                self.waiting_max = max(self.waiting_max, sample["vllm:num_requests_waiting"])
                self.kv_max = max(self.kv_max, sample["vllm:kv_cache_usage_perc"])
                self.samples += 1
            except Exception:
                self.errors += 1
            self.stop.wait(self.interval)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        self.thread.join()
        return False


def shift_times(rows, origin):
    shifted = []
    for row in rows:
        item = dict(row)
        item["t0_s"] = row["t0"] - origin
        item["t_done_s"] = row["t_done"] - origin
        item.pop("t0", None)
        item.pop("t_done", None)
        shifted.append(item)
    return shifted


def run_cell(spec, call, metrics_url):
    jobs = [{"i": index} for index in range(spec["requests"])]
    if spec["warmup"]:
        call({"i": -1}, warmup=True)
    before = scrape(metrics_url)
    origin = time.perf_counter()
    with GaugeSampler(metrics_url) as sampler:
        if spec["burst"]:
            raw = burst(jobs, lambda job: call(job, warmup=False))
        else:
            raw = closed_loop(jobs, spec["concurrency"], lambda job: call(job, warmup=False))
    after = scrape(metrics_url)
    rows = shift_times(raw, origin)
    return rows, before, after, sampler
