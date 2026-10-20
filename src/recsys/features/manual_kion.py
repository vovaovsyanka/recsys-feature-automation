"""KION: ручной набор признаков и пул грамматики для автоматического отбора.

Особенности домена: пара (пользователь, фильм) встречается один раз, поэтому прямых связей нет и ключ UI не используется;
косвенные связи - UA = пользователь×главный жанр и UB = пользователь×первый режиссёр.
Каналы: listen (любое взаимодействие), plus (>= 50% просмотра), skip (< 20%), full (>= 90%).
"""
from recsys import features as F
from recsys.features.grammar import HALFLIVES, WINDOWS
CHANS = ["listen", "plus", "skip", "full"]
KEYS = ["U", "I", "UA", "UB", "A", "B"]


def manual_groups() -> dict[str, list[str]]:
    G = {}
    G["user"] = (
        [f"U.listen.cnt.{w}" for w in ["1", "7", "30", "all"]] + [f"U.plus.cnt.{w}" for w in ["7", "30", "all"]] +
        [f"U.skip.cnt.{w}" for w in ["7", "30"]] + [f"U.listen.mean.{w}" for w in ["7", "30", "all"]] +
        [f"U.listen.sec.{w}" for w in ["7", "30", "all"]] + [f"U.listen.nuniq.{w}" for w in ["7", "30"]] +
        ["U.listen.last.all", "U.listen.first.all",
         "d:ratio(U.plus.cnt.7|U.listen.cnt.7)", "d:ratio(U.plus.cnt.30|U.listen.cnt.30)", "d:ratio(U.skip.cnt.30|U.listen.cnt.30)",
         "d:ratio(U.listen.cnt.7|U.listen.cnt.30)"])
    G["item"] = (
        [f"I.listen.cnt.{w}" for w in ["1", "7", "30", "all"]] + [f"I.plus.cnt.{w}" for w in ["7", "30"]] +
        [f"I.skip.cnt.{w}" for w in ["7", "30"]] + [f"I.listen.nuniq.{w}" for w in ["7", "30"]] +
        [f"I.listen.mean.{w}" for w in ["7", "30", "all"]] + ["I.listen.first.all", "I.listen.last.all"] +
        ["d:ratio(I.plus.cnt.7|I.listen.cnt.7)", "d:ratio(I.plus.cnt.30|I.listen.cnt.30)", "d:ratio(I.skip.cnt.30|I.listen.cnt.30)",
         "d:ratio(I.listen.cnt.1|I.listen.cnt.7)", "d:ratio(I.listen.cnt.7|I.listen.cnt.30)"])
    G["user_genre"] = (
        [f"UA.listen.cnt.{w}" for w in ["1", "7", "30", "all"]] + [f"UA.plus.cnt.{w}" for w in ["7", "30", "all"]] +
        [f"UA.skip.cnt.{w}" for w in ["30", "all"]] + [f"UA.listen.mean.{w}" for w in ["30", "all"]] +
        ["UA.listen.last.all", "UA.listen.first.all", "UA.listen.nuniq.30", "UA.listen.nuniq.all",
         "d:ratio(UA.listen.cnt.30|U.listen.cnt.30)", "d:ratio(UA.listen.cnt.all|U.listen.cnt.all)",
         "d:ratio(UA.plus.cnt.all|U.plus.cnt.all)", "d:urank(UA.listen.cnt.30)", "d:urank(UA.listen.cnt.all)",
         "d:urank(UA.listen.last.all)"])
    G["user_director"] = (
        [f"UB.listen.cnt.{w}" for w in ["30", "all"]] + ["UB.plus.cnt.all", "UB.listen.last.all", "UB.listen.mean.all",
                                                          "d:ratio(UB.listen.cnt.all|U.listen.cnt.all)", "d:urank(UB.listen.cnt.all)"])
    G["genre_director"] = (
        [f"A.listen.cnt.{w}" for w in ["7", "30"]] + [f"A.plus.cnt.{w}" for w in ["7", "30"]] + ["A.listen.nuniq.30"] +
        [f"B.listen.cnt.{w}" for w in ["7", "30"]] + ["B.listen.nuniq.30", "d:ratio(I.listen.cnt.30|A.listen.cnt.30)",
                                                      "d:ratio(I.listen.cnt.30|B.listen.cnt.30)"])
    G["cand"] = ["s1_rank", "s1_score", "s2_rank", "s2_score", "s3_rank", "s3_score"]
    return G


def manual_all() -> list[str]:
    return [n for g in manual_groups().values() for n in g]


def pool() -> list[str]:
    names = []
    for key in KEYS:
        for c in CHANS:
            names += [f"{key}.{c}.cnt.{w}" for w in WINDOWS]
            names += [f"{key}.{c}.decay.{h}" for h in HALFLIVES]
        names += [f"{key}.listen.mean.{w}" for w in WINDOWS]
        names += [f"{key}.listen.sec.{w}" for w in WINDOWS]
        names += [f"{key}.listen.last.{w}" for w in WINDOWS] + [f"{key}.plus.last.{w}" for w in WINDOWS]
        names += [f"{key}.listen.first.{w}" for w in WINDOWS]
        if key in F.PARTNER:
            names += [f"{key}.{c}.nuniq.{w}" for c in ["listen", "plus"] for w in WINDOWS]
    names = [n for n in names if F.Spec.parse(n).valid()]
    d = []
    for key in KEYS:
        for c in ["plus", "skip", "full"]:
            for w in ["7", "30", "all"]:
                d.append(f"d:ratio({key}.{c}.cnt.{w}|{key}.listen.cnt.{w})")
    for key in ["UA", "UB"]:
        for c in ["listen", "plus"]:
            for w in WINDOWS:
                d.append(f"d:ratio({key}.{c}.cnt.{w}|U.{c}.cnt.{w})")
            for w in ["1", "7", "14", "30", "90", "all"]:
                d.append(f"d:urank({key}.{c}.cnt.{w})")
        for h in ["h3", "h14", "h60"]:
            d.append(f"d:urank({key}.listen.decay.{h})")
        d.append(f"d:urank({key}.listen.last.all)")
    for (a, b) in [("I", "A"), ("I", "B"), ("UA", "UB")]:
        for w in WINDOWS:
            d.append(f"d:ratio({a}.listen.cnt.{w}|{b}.listen.cnt.{w})")
    for key in ["I", "U", "UA", "A"]:
        for (w1, w2) in [("1", "7"), ("7", "30"), ("30", "all"), ("3", "14")]:
            d.append(f"d:ratio({key}.listen.cnt.{w1}|{key}.listen.cnt.{w2})")
    return list(dict.fromkeys(names + d))

