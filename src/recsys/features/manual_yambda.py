"""Ручной набор признаков для Yambda: то, что написал бы практик. Окна фиксированы заранее (1, 7, 30 суток и вся история)
и не подбирались по val. Набор разбит на группы - это нужно для аблации.

Записан в той же грамматике, что и автоматические признаки, поэтому проходит те же проверки на утечки.
"""


def manual_groups() -> dict[str, list[str]]:
    G: dict[str, list[str]] = {}

    # ---- пользователь
    u = []
    u += [f"U.listen.cnt.{w}" for w in ["1", "7", "30", "all"]]
    u += [f"U.plus.cnt.{w}" for w in ["7", "30"]]
    u += [f"U.skip.cnt.{w}" for w in ["7", "30"]]
    u += [f"U.rec.cnt.{w}" for w in ["7", "30"]]
    u += [f"U.listen.mean.{w}" for w in ["7", "30"]]
    u += [f"U.listen.sec.{w}" for w in ["7", "30"]]
    u += [f"U.listen.nuniq.{w}" for w in ["7", "30"]]
    u += [f"U.like.cnt.{w}" for w in ["30", "all"]]
    u += [f"U.dislike.cnt.{w}" for w in ["30", "all"]]
    u += ["U.listen.last.all", "U.listen.first.all"]
    u += ["d:ratio(U.plus.cnt.7|U.listen.cnt.7)", "d:ratio(U.plus.cnt.30|U.listen.cnt.30)",
          "d:ratio(U.rec.cnt.7|U.listen.cnt.7)", "d:ratio(U.rec.cnt.30|U.listen.cnt.30)",
          "d:ratio(U.listen.nuniq.30|U.listen.cnt.30)"]
    G["user"] = u

    # ---- трек (глобальные)
    i = []
    i += [f"I.listen.cnt.{w}" for w in ["1", "7", "30"]]
    i += [f"I.plus.cnt.{w}" for w in ["1", "7", "30"]]
    i += [f"I.skip.cnt.{w}" for w in ["7", "30"]]
    i += [f"I.listen.nuniq.{w}" for w in ["7", "30"]]
    i += [f"I.listen.mean.{w}" for w in ["7", "30"]]
    i += [f"I.like.cnt.{w}" for w in ["30", "all"]]
    i += [f"I.dislike.cnt.{w}" for w in ["30", "all"]]
    i += ["I.listen.first.all", "I.listen.last.all"]
    i += ["d:ratio(I.plus.cnt.7|I.listen.cnt.7)", "d:ratio(I.plus.cnt.30|I.listen.cnt.30)",
          "d:ratio(I.skip.cnt.30|I.listen.cnt.30)", "d:ratio(I.listen.cnt.1|I.listen.cnt.7)",
          "d:ratio(I.like.cnt.30|I.listen.cnt.30)"]
    G["item"] = i

    # ---- пара пользователь-трек (прямые связи)
    ui = []
    ui += [f"UI.listen.cnt.{w}" for w in ["1", "7", "30", "all"]]
    ui += [f"UI.plus.cnt.{w}" for w in ["1", "7", "30", "all"]]
    ui += [f"UI.skip.cnt.{w}" for w in ["7", "30", "all"]]
    ui += [f"UI.listen.mean.{w}" for w in ["30", "all"]]
    ui += [f"UI.listen.sec.{w}" for w in ["30", "all"]]
    ui += ["UI.listen.last.all", "UI.plus.last.all", "UI.listen.first.all"]
    ui += ["UI.like.cnt.all", "UI.unlike.cnt.all", "UI.dislike.cnt.all", "UI.undislike.cnt.all"]
    ui += ["d:diff(UI.like.cnt.all|UI.unlike.cnt.all)",
           "d:ratio(UI.listen.cnt.30|U.listen.cnt.30)", "d:ratio(UI.listen.cnt.all|U.listen.cnt.all)",
           "d:ratio(UI.skip.cnt.all|UI.listen.cnt.all)", "d:ratio(UI.plus.cnt.all|UI.listen.cnt.all)",
           "d:urank(UI.plus.cnt.7)", "d:urank(UI.plus.cnt.30)", "d:urank(UI.plus.cnt.all)", "d:urank(UI.listen.last.all)"]
    G["user_item"] = ui

    # ---- косвенные связи: артист и альбом
    ua = []
    ua += [f"UA.listen.cnt.{w}" for w in ["1", "7", "30", "all"]]
    ua += [f"UA.plus.cnt.{w}" for w in ["7", "30", "all"]]
    ua += [f"UA.skip.cnt.{w}" for w in ["30", "all"]]
    ua += [f"UA.listen.mean.{w}" for w in ["30", "all"]]
    ua += ["UA.listen.last.all", "UA.listen.first.all", "UA.like.cnt.all", "UA.listen.nuniq.30", "UA.listen.nuniq.all"]
    ua += ["d:ratio(UA.listen.cnt.30|U.listen.cnt.30)", "d:ratio(UA.listen.cnt.all|U.listen.cnt.all)",
           "d:ratio(UI.listen.cnt.all|UA.listen.cnt.all)", "d:urank(UA.plus.cnt.30)"]
    ub = [f"UB.listen.cnt.{w}" for w in ["7", "30", "all"]] + ["UB.plus.cnt.30", "UB.plus.cnt.all", "UB.listen.last.all",
                                                              "d:ratio(UB.listen.cnt.30|U.listen.cnt.30)"]
    G["user_artist"] = ua
    G["user_album"] = ub

    # ---- глобальный артист
    a = [f"A.listen.cnt.{w}" for w in ["7", "30"]] + [f"A.plus.cnt.{w}" for w in ["7", "30"]] + \
        [f"A.listen.nuniq.{w}" for w in ["7", "30"]] + ["d:ratio(I.listen.cnt.30|A.listen.cnt.30)"]
    G["artist"] = a

    # ---- признаки источников кандидатов (ранги/оценки из кандидатной стадии; считаются по истории до якоря)
    G["cand"] = ["s1_rank", "s1_score", "s2_rank", "s2_score", "s3_rank", "s3_score", "s4_rank", "s4_score"]
    return G


def manual_all(exclude: tuple[str, ...] = ()) -> list[str]:
    return [n for group, names in manual_groups().items() if group not in exclude for n in names]

