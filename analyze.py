"""
analyze.py
수집한 AI 응답에서 브랜드 언급을 추출하고 노출 지표를 산출한다.

지표 정의
---------
  SoV (노출 점유율)  : 해당 브랜드가 언급된 응답 비율
  Top1 점유율        : 응답에서 가장 먼저 언급된 비율 (= 1순위 추천 자리)
  평균 언급 순위     : 언급됐을 때 몇 번째로 등장하는지 (낮을수록 앞자리)

방법론 주의
-----------
  질의문 자체가 브랜드를 지목한 경우(예: "LG랑 삼성 뭐가 나아?")
  그 브랜드의 언급은 '유도된 언급'이므로 SoV 집계에서 제외한다.
  제외 건수는 리포트에 함께 기록한다.
"""

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
PROMPTS = ROOT / "data" / "prompts.json"
RESPONSES = ROOT / "data" / "responses.jsonl"
OUTDIR = ROOT / "out"


def build_matcher(aliases: dict) -> dict:
    """brand -> 컴파일된 정규식. 별칭이 긴 것부터 매칭되도록 정렬."""
    matchers = {}
    for brand, alts in aliases.items():
        terms = sorted(set([brand] + alts), key=len, reverse=True)
        pattern = "|".join(re.escape(t) for t in terms)
        matchers[brand] = re.compile(pattern, re.IGNORECASE)
    return matchers


def first_positions(text: str, matchers: dict) -> dict:
    """각 브랜드가 텍스트에서 '처음' 등장하는 위치. 없으면 제외."""
    pos = {}
    for brand, rx in matchers.items():
        m = rx.search(text)
        if m:
            pos[brand] = m.start()
    return pos


def main():
    payload = json.loads(PROMPTS.read_text(encoding="utf-8"))
    cat_meta = payload["meta"]["categories"]

    matchers = {cat: build_matcher(meta["aliases"]) for cat, meta in cat_meta.items()}

    # 1차 통과: 조건별 응답 수를 세어 '토막 데이터'를 걸러낸다.
    # 중간에 쿼터로 끊긴 조건이 소수 건수로 남아 있으면 점유율이 왜곡되므로,
    # 가장 많이 모인 조건의 절반에 못 미치는 조건은 분석에서 제외한다.
    counts = {}
    with RESPONSES.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("ok") and rec.get("text"):
                counts[rec["condition"]] = counts.get(rec["condition"], 0) + 1
    if not counts:
        raise SystemExit("분석할 응답이 없습니다. collect.py를 먼저 실행하세요.")

    cutoff = max(counts.values()) * 0.5
    keep = {c for c, n in counts.items() if n >= cutoff}
    dropped = {c: n for c, n in counts.items() if c not in keep}
    if dropped:
        print(f"[제외] 응답 수가 부족한 조건: {dropped} (기준 {cutoff:.0f}건 미만)")

    rows = []
    n_resp = 0
    n_excluded = 0

    with RESPONSES.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not rec.get("ok") or not rec.get("text"):
                continue
            if rec["condition"] not in keep:
                continue

            n_resp += 1
            cat = rec["category"]
            mx = matchers[cat]

            # 질의문이 직접 지목한 브랜드 → 유도된 언급이므로 제외 대상
            prompted = set(first_positions(rec["prompt"], mx).keys())

            pos = first_positions(rec["text"], mx)
            ranked = sorted(pos.items(), key=lambda kv: kv[1])

            for rank, (brand, _idx) in enumerate(ranked, start=1):
                if brand in prompted:
                    n_excluded += 1
                    continue
                rows.append(
                    {
                        "prompt_id": rec["prompt_id"],
                        "category": cat,
                        "intent": rec["intent"],
                        "condition": rec["condition"],
                        "run": rec["run"],
                        "brand": brand,
                        "rank": rank,
                        "is_top1": rank == 1,
                    }
                )

            # 아무 브랜드도 언급 안 된 응답도 분모에 들어가야 하므로 기록
            rows.append(
                {
                    "prompt_id": rec["prompt_id"],
                    "category": cat,
                    "intent": rec["intent"],
                    "condition": rec["condition"],
                    "run": rec["run"],
                    "brand": "__RESPONSE__",
                    "rank": 0,
                    "is_top1": False,
                }
            )

    if not rows:
        raise SystemExit("분석할 응답이 없습니다. collect.py를 먼저 실행하세요.")

    df = pd.DataFrame(rows)
    OUTDIR.mkdir(exist_ok=True)
    df.to_csv(OUTDIR / "mentions.csv", index=False, encoding="utf-8-sig")

    # 분모: 응답 수 (카테고리·조건·의도별)
    denom = (
        df[df.brand == "__RESPONSE__"]
        .groupby(["category", "condition", "intent"])
        .size()
        .rename("n_responses")
    )
    m = df[df.brand != "__RESPONSE__"]

    # ---- 1) 전체 노출 점유율 (카테고리 × 조건) -----------------------------
    d2 = denom.groupby(["category", "condition"]).sum()
    sov = (
        m.groupby(["category", "condition", "brand"])
        .agg(mentions=("brand", "size"),
             top1=("is_top1", "sum"),
             mean_rank=("rank", "mean"))
        .join(d2)
        .assign(
            sov=lambda x: (x.mentions / x.n_responses * 100).round(1),
            top1_share=lambda x: (x.top1 / x.n_responses * 100).round(1),
            mean_rank=lambda x: x.mean_rank.round(2),
        )
        .sort_values(["category", "condition", "sov"], ascending=[True, True, False])
        .reset_index()
    )
    sov.to_csv(OUTDIR / "sov_by_condition.csv", index=False, encoding="utf-8-sig")

    # ---- 2) 의도별 노출 점유율 --------------------------------------------
    sov_intent = (
        m.groupby(["category", "intent", "brand"])
        .agg(mentions=("brand", "size"), top1=("is_top1", "sum"))
        .join(denom.groupby(["category", "intent"]).sum())
        .assign(
            sov=lambda x: (x.mentions / x.n_responses * 100).round(1),
            top1_share=lambda x: (x.top1 / x.n_responses * 100).round(1),
        )
        .sort_values(["category", "intent", "sov"], ascending=[True, True, False])
        .reset_index()
    )
    sov_intent.to_csv(OUTDIR / "sov_by_intent.csv", index=False, encoding="utf-8-sig")

    # ---- 3) 조건(엔진)이 바뀌면 노출이 얼마나 달라지는가 --------------------
    #        같은 질문·같은 브랜드인데 엔진만 바꿨을 때의 점유율 차이.
    #        전통 매체로 치면 매체사별 도달 차이를 보는 것과 같은 구조다.
    pivot = sov.pivot_table(index=["category", "brand"], columns="condition", values="sov")
    conds = [c for c in ["internal", "internal_alt", "grounded"] if c in pivot.columns]
    if len(conds) >= 2:
        a, b = conds[0], conds[1]
        pivot["delta"] = (pivot[b] - pivot[a]).round(1)
        pivot = pivot.sort_values("delta", ascending=False)
        print(f"\n[조건 비교] {a} → {b} 노출 점유율 변화 상·하위")
        pv = pivot.reset_index()
        for label, part in (("증가", pv.head(3)), ("감소", pv.tail(3))):
            items = ", ".join(f"{r.brand}({r.category}) {r.delta:+.1f}%p" for r in part.itertuples())
            print(f"  {label}: {items}")
    pivot.reset_index().to_csv(OUTDIR / "condition_effect.csv", index=False, encoding="utf-8-sig")

    print(f"응답 {n_resp}건 분석 / 유도된 언급 {n_excluded}건 제외")
    print(f"결과 저장: {OUTDIR}/")
    for cat in cat_meta:
        print(f"\n[{cat}] 조건별 상위 노출 브랜드")
        sub = sov[sov.category == cat]
        for cond in sub.condition.unique():
            top = sub[sub.condition == cond].head(5)
            line = ", ".join(f"{r.brand} {r.sov}%" for r in top.itertuples())
            print(f"  {cond:9s} {line}")


if __name__ == "__main__":
    main()
