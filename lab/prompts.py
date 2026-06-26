import hashlib

from lab.settings import load_config


PLACES = ("зал", "полка", "каталог", "карточка", "журнал", "шифр", "формуляр", "опись")
ACTIONS = ("хранит", "указывает", "повторяет", "сверяет", "оставляет", "нумерует")


def sentence(index):
    place = PLACES[index % len(PLACES)]
    action = ACTIONS[(index // len(PLACES)) % len(ACTIONS)]
    number = index + 1
    return (
        f"Строка {number}: открытый {place} {action} запись {number}, "
        f"и дата выдачи остаётся в журнале читального зала номер {number}."
    )


def corpus_lines(count):
    return [sentence(index) for index in range(count)]


def system_text(kind, index, lines):
    config = load_config()
    short_n = config["short_lines"]
    long_n = config["long_lines"]
    if len(lines) < long_n:
        raise ValueError(f"в корпусе {len(lines)} строк, для длинного префикса нужно {long_n}")
    body = list(lines[:long_n])
    if kind == "short":
        return "\n".join(body[:short_n])
    if kind == "hit":
        return "\n".join(body)
    if kind == "cold":
        body[0] = sentence(50_000)
        return "\n".join(body)
    if kind == "miss":
        body[0] = sentence(20_000 + index)
        return "\n".join(body)
    if kind == "queue":
        body = list(lines[:short_n])
        body[0] = sentence(30_000 + index)
        return "\n".join(body)
    raise ValueError(f"неизвестный префикс {kind}")


def user_text(index):
    config = load_config()
    return f"{config['user_prompt']} Номер {index}."


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
