import json
import time
import urllib.error
import urllib.request


def parse_sse_line(line):
    text = line.strip()
    if not text.startswith("data:"):
        return "skip", None
    body = text[5:].strip()
    if body == "[DONE]":
        return "done", None
    if not body:
        return "skip", None
    return "event", json.loads(body)


def note_event(state, event, now):
    if event.get("usage"):
        state["usage"] = event["usage"]
    choices = event.get("choices") or []
    if not choices:
        return
    choice = choices[0]
    if choice.get("finish_reason"):
        state["finish_reason"] = choice["finish_reason"]
    delta = choice.get("delta") or {}
    piece = delta.get("content") or ""
    if not piece:
        return
    state["chunks"] += 1
    if state["t_first"] is None:
        state["t_first"] = now
    state["t_last"] = now
    if len(state["text"]) < 80:
        state["text"] = (state["text"] + piece)[:80]


def empty_state():
    return {
        "t_first": None,
        "t_last": None,
        "chunks": 0,
        "usage": None,
        "finish_reason": None,
        "text": "",
    }


class StreamClient:
    def __init__(self, base_url, model, timeout, temperature, seed, max_tokens):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens

    def stream(self, system, user, max_tokens=None):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "seed": self.seed,
            "stream": True,
            "ignore_eos": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=data,
            headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        )
        t0 = time.perf_counter()
        state = empty_state()
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            while True:
                raw = response.readline()
                if not raw:
                    break
                kind, event = parse_sse_line(raw.decode("utf-8", errors="replace"))
                now = time.perf_counter()
                if kind == "done":
                    break
                if kind == "event":
                    note_event(state, event, now)
        t_end = time.perf_counter()
        usage = state["usage"] or {}
        completion = int(usage.get("completion_tokens") or 0)
        prompt = int(usage.get("prompt_tokens") or 0)
        if state["t_first"] is None:
            raise RuntimeError("поток не прислал ни одного токена")
        from lab.metrics import tpot

        return {
            "ok": True,
            "t0": t0,
            "t_done": t_end,
            "ttft_s": state["t_first"] - t0,
            "tpot_s": tpot(state["t_first"], state["t_last"], completion),
            "e2e_s": t_end - t0,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "chunks": state["chunks"],
            "finish_reason": state["finish_reason"],
            "text": state["text"],
        }


def discover_model(base_url, timeout=15):
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    request = urllib.request.Request(root + "/v1/models")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    rows = body.get("data") or []
    if not rows:
        raise RuntimeError("сервер не вернул ни одной модели")
    return rows[0]


def tokenize(base_url, model, text, timeout=60):
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    data = json.dumps({"model": model, "prompt": text}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        root + "/tokenize",
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"tokenize HTTP {exc.code}: {detail[:300]}") from exc
    return int(body["count"])
