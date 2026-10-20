"""Пространство поиска автоматических методов: пул признаков, который порождает грамматика.

Пул задаёт, из чего методы отбора вообще могут выбирать. Он содержит все допустимые комбинации
(ключ, канал, агрегат, окно), а также производные признаки: доли, тренды окон, ранги внутри пользователя.
"""
from recsys import features as F

WINDOWS = ["1", "2", "3", "5", "7", "10", "14", "21", "30", "45", "60", "90", "all"]
HALFLIVES = ["h1", "h2", "h3", "h5", "h7", "h10", "h14", "h21", "h30", "h45", "h60", "h90"]
DECAY_CHANS = ["listen", "plus", "skip", "full", "org", "rec", "like", "dislike"]


def _base_specs() -> list[str]:
    names = []
    for key in F.KEYS:
        for chan in F.CHANS:
            names += [f"{key}.{chan}.cnt.{w}" for w in WINDOWS]
        for chan in DECAY_CHANS:
            names += [f"{key}.{chan}.decay.{h}" for h in HALFLIVES]
        for chan in F.SR_CH:
            names += [f"{key}.{chan}.mean.{w}" for w in WINDOWS]
        names += [f"{key}.listen.sec.{w}" for w in WINDOWS]
        for chan in F.TMAX_CH:
            names += [f"{key}.{chan}.last.{w}" for w in WINDOWS]
        for chan in F.TMIN_CH:
            names += [f"{key}.{chan}.first.{w}" for w in WINDOWS]
        if key in F.PARTNER:
            for chan in ["listen", "plus"]:
                names += [f"{key}.{chan}.nuniq.{w}" for w in WINDOWS]
    return [n for n in names if F.Spec.parse(n).valid()]


def _derived_specs() -> list[str]:
    d = []
    # конверсии канала в прослушивание в том же окне
    for key in F.KEYS:
        for chan in ["plus", "skip", "full", "org", "rec", "like"]:
            for w in ["7", "30", "all"]:
                d.append(f"d:ratio({key}.{chan}.cnt.{w}|{key}.listen.cnt.{w})")
    # доля пары/артиста/альбома в активности пользователя
    for key in ["UI", "UA", "UB"]:
        for chan in ["listen", "plus"]:
            for w in WINDOWS:
                d.append(f"d:ratio({key}.{chan}.cnt.{w}|U.{chan}.cnt.{w})")
    # ранги среди кандидатов того же пользователя
    for key in ["UI", "UA", "UB"]:
        for chan in ["listen", "plus"]:
            for w in ["1", "3", "7", "14", "30", "90", "all"]:
                d.append(f"d:urank({key}.{chan}.cnt.{w})")
    for key in ["UI", "UA"]:
        for h in ["h1", "h3", "h7", "h14", "h30", "h90"]:
            d.append(f"d:urank({key}.plus.decay.{h})")
    for n in ["UI.listen.last.all", "UI.plus.last.all", "UA.listen.last.all", "UB.listen.last.all"]:
        d.append(f"d:urank({n})")
    # отношения между ключами (пара/артист, артист/пользователь, трек/артист, артист/альбом)
    for a, b in [("UI", "UA"), ("UA", "U"), ("I", "A"), ("UA", "UB")]:
        for chan in ["listen", "plus"]:
            for w in WINDOWS:
                d.append(f"d:ratio({a}.{chan}.cnt.{w}|{b}.{chan}.cnt.{w})")
    # тренд: короткое окно к длинному
    for key in ["I", "U", "UI", "UA", "A"]:
        for chan in ["listen", "plus"]:
            for w1, w2 in [("1", "7"), ("7", "30"), ("30", "all"), ("3", "14")]:
                d.append(f"d:ratio({key}.{chan}.cnt.{w1}|{key}.{chan}.cnt.{w2})")
    for key in ["UI", "UA", "I", "U"]:
        d.append(f"d:diff({key}.like.cnt.all|{key}.unlike.cnt.all)")
    # разнообразие: число уникальных треков на число прослушиваний
    for key in ["U", "UA", "I"]:
        for w in ["7", "30", "all"]:
            d.append(f"d:ratio({key}.listen.nuniq.{w}|{key}.listen.cnt.{w})")
    return d


def grammar_pool(include_derived: bool = True) -> list[str]:
    """Полный пул признаков Yambda (порядка 3,6 тыс.)."""
    names = _base_specs() + (_derived_specs() if include_derived else [])
    return list(dict.fromkeys(names))
