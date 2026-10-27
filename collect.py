"""
collect.py
AI 검색 엔진에 질의를 던지고 응답을 수집한다.

두 개 이상의 '조건'을 비교 수집하는 것이 핵심 설계다.
조건이란 = 어떤 엔진에게, 어떤 방식으로 물었는가.

  internal      : gemini-2.5-flash, 모델 내부 지식만으로 답변
  internal_alt  : gemini-2.0-flash, 모델 내부 지식만으로 답변
  grounded      : gemini-2.5-flash + Google Search 연결

기본값은 internal + internal_alt (둘 다 무료 티어).
두 모델의 차이가 곧 "엔진이 다르면 브랜드 도달이 달라지는가"다.
전통 매체로 치면 매체사별 도달 차이를 보는 것과 같은 구조의 질문이다.

grounded는 무료 티어에서 검색 연결 쿼터가 없어 429가 난다.
결제를 연결하면 쓸 수 있으나 필수는 아니다 — 핵심 가설(의도별 노출 편차)은
internal 계열만으로 완전히 검증된다. 못 쓴 사실은 리포트에 한계로 명시한다.

LLM 응답은 확률적이라 같은 질문도 답이 흔들린다.
따라서 각 (질의 × 조건)을 N회 반복 수집해 평균으로 본다.

사용법
------
  export GEMINI_API_KEY="발급받은키"
  python3 collect.py                      # 기본: internal + internal_alt
  python3 collect.py --status             # 지금까지 뭐가 모였는지 확인
  python3 collect.py --list-models        # 내 키로 쓸 수 있는 모델 확인
  python3 collect.py --conditions internal --limit 4   # 소량 테스트

중단되어도 이어받기 가능 (responses.jsonl에 이미 있는 건 건너뜀).
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import requests

BASE = "https://generativelanguage.googleapis.com/v1beta"
ROOT = Path(__file__).parent
PROMPTS = ROOT / "data" / "prompts.json"
OUTFILE = ROOT / "data" / "responses.jsonl"

# 조건 정의: 이름 -> (검색 연결 여부, 선호 모델 후보)
#
# 모델 이름을 하드코딩하지 않는다. 구글이 모델을 자주 폐기하고
# 계정마다 접근 가능한 모델이 다르기 때문에, 실행 시점에
# 내 키로 실제 쓸 수 있는 목록을 받아서 자동으로 고른다.
CONDITIONS = {
    "internal":     {"grounded": False, "prefer": ["flash", "pro"]},
    "internal_alt": {"grounded": False, "prefer": ["lite", "pro", "flash"]},
    "grounded":     {"grounded": True,  "prefer": ["flash", "pro"]},
}
DEFAULT_CONDITIONS = "internal,internal_alt"

# 채팅용이 아닌 모델은 후보에서 제외
EXCLUDE = ("embedding", "imagen", "veo", "tts", "audio", "image", "live",
           "aqa", "learnlm", "vision")


def load_dotenv():
    """같은 폴더의 .env를 읽어 환경변수로 올린다 (외부 패키지 없이).
    export를 매번 치지 않아도 되게 하기 위한 편의 기능."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().removeprefix("export ").strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def api_key() -> str:
    load_dotenv()
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        sys.exit(
            "GEMINI_API_KEY를 찾을 수 없습니다.\n"
            "  https://aistudio.google.com/apikey 에서 무료 발급 후, 둘 중 하나로 넣으세요:\n"
            '    1) 같은 폴더에 .env 파일:  GEMINI_API_KEY=키값\n'
            '    2) 터미널에서:             export GEMINI_API_KEY="키값"'
        )

    # 키 형식은 검사하지 않는다.
    # 구글은 기존 'AIza...' 키를 'AQ.Ab...' 형식으로 전환했고(2026),
    # 둘 다 네이티브 Gemini 엔드포인트에서는 정상 동작한다.
    # 형식으로 키를 판단하면 오진이 난다.
    return key


def scrub(text: str, key: str) -> str:
    """에러 메시지에 키가 섞여 나오는 것을 막는다."""
    return text.replace(key, "***KEY***") if key else text


def auth_headers(key: str) -> dict:
    """인증은 헤더로 보낸다.
    ?key=... 쿼리 파라미터 방식도 동작하지만, 실패 시 스택트레이스의 URL에
    키가 그대로 찍힌다. 헤더로 보내면 로그에 키가 남지 않는다."""
    return {"x-goog-api-key": key, "Content-Type": "application/json"}


def fetch_models(key: str) -> list:
    """내 키로 generateContent를 쓸 수 있는 모델 이름 목록."""
    try:
        r = requests.get(f"{BASE}/models", headers=auth_headers(key), timeout=30)
    except requests.exceptions.RequestException as e:
        sys.exit(
            "네트워크로 Google API에 연결하지 못했습니다.\n"
            "  이 스크립트는 인터넷이 열린 환경(예: VSCode 터미널)에서 실행해야 합니다.\n"
            f"  상세: {scrub(str(e), key)[:200]}"
        )
    if r.status_code != 200:
        sys.exit(f"API 오류 HTTP {r.status_code}: {scrub(r.text, key)[:300]}")

    names = []
    for m in r.json().get("models", []):
        if "generateContent" not in m.get("supportedGenerationMethods", []):
            continue
        name = m["name"].replace("models/", "")
        if any(x in name.lower() for x in EXCLUDE):
            continue
        names.append(name)
    return sorted(names)


def list_models():
    for n in fetch_models(api_key()):
        print(f"  {n}")


def probe(model: str, key: str, cache: dict) -> bool:
    """모델을 실제로 한 번 호출해 보고 쓸 수 있는지 확인한다.

    /models 목록에 올라와 있어도 계정에 따라 404가 나는 모델이 있다
    ("no longer available to new users"). 목록만 믿으면 본 수집 도중에
    터지므로, 아주 짧은 호출로 미리 검증한다."""
    if model in cache:
        return cache[model]
    try:
        r = requests.post(
            f"{BASE}/models/{model}:generateContent",
            headers=auth_headers(key),
            json={"contents": [{"parts": [{"text": "hi"}]}],
                  "generationConfig": {"maxOutputTokens": 8}},
            timeout=45,
        )
        ok = r.status_code == 200
    except requests.exceptions.RequestException:
        ok = False
    cache[model] = ok
    return ok


def _version(name: str) -> float:
    """'gemini-2.5-flash' -> 2.5, 'gemini-3-pro' -> 3.0. 최신 우선 정렬용."""
    m = re.search(r"gemini-(\d+)(?:\.(\d+))?", name)
    if not m:
        return 0.0
    return float(f"{m.group(1)}.{m.group(2) or 0}")


def resolve_models(conditions: list, key: str, override: str | None) -> dict:
    """조건별로 실제 사용할 모델을 결정한다.
    후보를 '최신 → 안정판 우선' 순으로 세운 뒤, 실제 호출이 성공하는
    첫 모델을 고른다. 조건끼리는 서로 다른 모델이 되게 한다
    (같은 모델이면 '엔진 간 비교'라는 축 자체가 성립하지 않으므로).
    """
    if override:
        return {c: override for c in conditions}

    avail = fetch_models(key)
    if not avail:
        sys.exit("사용 가능한 모델이 없습니다. 키를 확인하세요.")

    # 최신 버전 우선, 같은 버전이면 안정판(preview/exp 아닌 것) 우선
    avail.sort(key=lambda n: (
        -_version(n),
        any(x in n for x in ("preview", "exp", "latest")),
        len(n),
    ))

    print(f"모델 {len(avail)}개 중에서 실제 호출 가능한 것을 확인합니다...")
    resolved, used, cache = {}, set(), {}

    for cond in conditions:
        # 선호 키워드 순 → 그래도 없으면 전체
        ordered = []
        for kw in CONDITIONS[cond]["prefer"]:
            ordered += [n for n in avail if kw in n.lower() and n not in ordered]
        ordered += [n for n in avail if n not in ordered]

        pick = None
        for n in ordered:
            if n in used:
                continue
            if probe(n, key, cache):
                pick = n
                break
        if not pick:
            sys.exit(
                f"'{cond}' 조건에 쓸 수 있는 모델을 찾지 못했습니다.\n"
                f"  확인해 본 모델 {len(cache)}개가 모두 실패했습니다.\n"
                "  python3 collect.py --list-models 결과를 공유해 주세요."
            )
        resolved[cond] = pick
        used.add(pick)

    tried = sum(1 for v in cache.values() if not v)
    print(f"자동 선택 완료 (사용 불가 {tried}개 건너뜀):")
    for c, m in resolved.items():
        print(f"  {c:14s} -> {m}")
    return resolved


def ask(prompt: str, model: str, grounded: bool, key: str) -> dict:
    """Gemini에 1회 질의. 성공하면 {text, sources}, 실패하면 예외."""
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        # temperature를 0으로 고정하지 않는다.
        # 실제 소비자가 받는 답변의 변동성까지 포함해서 보는 게 목적이므로
        # 기본 샘플링을 유지하고 반복 측정으로 평균을 낸다.
        "generationConfig": {"maxOutputTokens": 1200},
    }
    if grounded:
        body["tools"] = [{"google_search": {}}]

    # 503(과부하)·5xx·타임아웃은 일시적 장애다.
    # 그 자리에서 몇 번 다시 시도해야 데이터에 구멍이 안 생긴다.
    url = f"{BASE}/models/{model}:generateContent"
    TRIES = 4
    for attempt in range(1, TRIES + 1):
        try:
            r = requests.post(url, headers=auth_headers(key), json=body, timeout=90)
        except requests.exceptions.RequestException as e:
            if attempt == TRIES:
                raise RuntimeError(f"네트워크 실패: {scrub(str(e), key)[:200]}")
            time.sleep(min(30, 3 * 2 ** (attempt - 1)) + random.random() * 2)
            continue

        if r.status_code == 200:
            break
        if r.status_code in (500, 502, 503, 504) and attempt < TRIES:
            wait = min(30, 3 * 2 ** (attempt - 1)) + random.random() * 2
            print(f"    · {r.status_code} 과부하, {wait:.0f}초 후 재시도 ({attempt}/{TRIES - 1})")
            time.sleep(wait)
            continue
        raise RuntimeError(f"HTTP {r.status_code}: {scrub(r.text, key)[:300]}")

    data = r.json()
    cand = (data.get("candidates") or [{}])[0]
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts)

    # grounding으로 실제 참조된 출처 도메인 (있으면 기록)
    sources = []
    gm = cand.get("groundingMetadata") or {}
    for chunk in gm.get("groundingChunks", []) or []:
        web = chunk.get("web") or {}
        if web.get("title"):
            sources.append(web["title"])

    return {"text": text, "sources": sources}


def load_done() -> set:
    done = set()
    if OUTFILE.exists():
        with OUTFILE.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("ok"):
                    done.add((rec["prompt_id"], rec["condition"], rec["run"]))
    return done


def show_status():
    """지금까지 수집된 현황을 조건별로 요약."""
    if not OUTFILE.exists():
        print("아직 수집된 응답이 없습니다.")
        return
    from collections import Counter
    ok, fail = Counter(), Counter()
    for line in OUTFILE.open(encoding="utf-8"):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        (ok if rec.get("ok") else fail)[rec.get("condition", "?")] += 1

    n_prompts = len(json.loads(PROMPTS.read_text(encoding="utf-8"))["prompts"])
    print(f"질의 {n_prompts}개 기준 수집 현황")
    for cond in sorted(set(ok) | set(fail)):
        target = n_prompts * 3
        print(f"  {cond:14s} 성공 {ok[cond]:4d} / 목표 {target}   실패 {fail[cond]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="조건별 기본 모델을 덮어씀")
    ap.add_argument("--repeats", type=int, default=3, help="조건별 반복 횟수")
    ap.add_argument("--limit", type=int, default=0, help="질의 개수 제한(테스트용)")
    ap.add_argument("--sleep", type=float, default=4.0, help="호출 간 대기(초)")
    ap.add_argument("--conditions", default=DEFAULT_CONDITIONS)
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.list_models:
        list_models()
        return
    if args.status:
        show_status()
        return

    key = api_key()
    payload = json.loads(PROMPTS.read_text(encoding="utf-8"))
    prompts = payload["prompts"]
    if args.limit:
        prompts = prompts[: args.limit]

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        sys.exit(f"알 수 없는 조건: {unknown} / 사용 가능: {list(CONDITIONS)}")
    models = resolve_models(conditions, key, args.model)
    done = load_done()

    jobs = [
        (p, cond, run)
        for p in prompts
        for cond in conditions
        for run in range(1, args.repeats + 1)
        if (p["prompt_id"], cond, run) not in done
    ]
    total = len(prompts) * len(conditions) * args.repeats
    print(f"전체 {total}건 / 이미 수집 {total - len(jobs)}건 / 이번에 {len(jobs)}건")

    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    fails = 0
    dead = set()  # 쿼터가 없어 포기한 조건

    with OUTFILE.open("a", encoding="utf-8") as out:
        for i, (p, cond, run) in enumerate(jobs, 1):
            if cond in dead:
                continue
            spec = CONDITIONS[cond]
            model = models[cond]
            rec = {
                "prompt_id": p["prompt_id"],
                "category": p["category"],
                "intent": p["intent"],
                "prompt": p["prompt"],
                "condition": cond,
                "run": run,
                "model": model,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            try:
                res = ask(p["prompt"], model, spec["grounded"], key)
                rec.update(ok=True, text=res["text"], sources=res["sources"])
                fails = 0
            except Exception as e:  # rate limit 포함
                msg = str(e)
                rec.update(ok=False, error=msg[:600], text="", sources=[])
                fails += 1

                # 429 응답에는 보통 retryDelay가 들어 있다. 그 값을 그대로 존중한다.
                # (분당 한도면 몇십 초, 일일 한도면 매우 큰 값이 온다)
                m = re.search(r'"?retryDelay"?\s*:\s*"?(\d+(?:\.\d+)?)s', msg)
                back = float(m.group(1)) + 1 if m else min(60, 5 * fails) + random.random() * 3
                print(f"  ! 실패({fails}회 연속) {back:.0f}초 대기 — {msg[:120]}")

                if back > 300:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    sys.exit(
                        f"\n'{model}'의 할당량이 소진됐습니다 (재시도 권장 대기 {back:.0f}초).\n"
                        "  일일 한도일 가능성이 큽니다. 다른 모델로 이어서 받으세요:\n"
                        "    python3 collect.py --conditions <조건> --model <다른모델>\n"
                        "  쓸 수 있는 모델: python3 collect.py --list-models"
                    )

                # 404는 모델이 폐기된 것 — 재시도해도 절대 안 되므로 즉시 중단
                if "404" in str(e):
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    sys.exit(
                        f"\n모델 '{model}'을(를) 쓸 수 없습니다 (404).\n"
                        "  python3 collect.py --list-models 로 목록을 확인한 뒤\n"
                        "  python3 collect.py --model <모델이름> 으로 지정해 주세요."
                    )

                # 한 조건이 계속 429면 쿼터 자체가 없는 것이다.
                # 몇 시간을 헛돌지 않도록 그 조건을 통째로 건너뛴다.
                if fails >= 8 and "429" in str(e):
                    print(f"  >> '{cond}' 조건은 쿼터가 없는 것으로 판단, 건너뜁니다.")
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    dead.add(cond)
                    fails = 0
                    continue

                time.sleep(back)

            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()

            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} 완료")
            time.sleep(args.sleep)

    print(f"저장: {OUTFILE}")


if __name__ == "__main__":
    main()
