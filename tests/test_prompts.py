from lab.prompts import corpus_lines
from lab.prompts import system_text
from lab.report import cell_plan
from lab.settings import load_config
from lab.settings import root


def test_miss_cells_use_disjoint_indexes():
    spans = []
    for spec in cell_plan():
        if spec["prefix"] != "miss":
            continue
        base = spec["index_base"]
        spans.append(set(range(base, base + spec["requests"] + 1)))
    assert spans
    for left in range(len(spans)):
        for right in range(left + 1, len(spans)):
            assert spans[left].isdisjoint(spans[right])


def test_prefix_file_matches_generator():
    config = load_config()
    text = (root() / "data" / "prefix.txt").read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line]
    assert lines == corpus_lines(config["long_lines"])


def test_corpus_is_deterministic_and_addressed():
    config = load_config()
    lines = corpus_lines(config["long_lines"])
    assert lines[0].startswith("Строка 1:")
    assert len(lines) == config["long_lines"]
    assert len(set(lines)) == len(lines)


def test_hit_shares_prefix_and_miss_changes_the_first_line():
    config = load_config()
    lines = corpus_lines(config["long_lines"])
    hit_a = system_text("hit", 1, lines)
    hit_b = system_text("hit", 2, lines)
    assert hit_a == hit_b
    miss = system_text("miss", 3, lines)
    assert miss.splitlines()[0] != hit_a.splitlines()[0]
    assert miss.splitlines()[1:] == hit_a.splitlines()[1:]
    cold_a = system_text("cold", 1, lines)
    cold_b = system_text("cold", 2, lines)
    assert cold_a == cold_b
    assert cold_a != hit_a
    short = system_text("short", 0, lines)
    assert short.splitlines() == hit_a.splitlines()[: config["short_lines"]]
    queued = system_text("queue", 4, lines)
    assert len(queued.splitlines()) == config["short_lines"]
    assert queued.splitlines()[0] != short.splitlines()[0]
