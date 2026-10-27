# AI 검색 브랜드 노출 분석

소비자가 AI 검색(Gemini 등)에 던질 법한 구매 탐색 질문을 설계해 응답을 수집하고, 응답 속 브랜드 언급을 파싱해 **구매 의도별 브랜드 노출 점유율(SoV)** 을 측정하는 프로젝트입니다.

핵심 질문: *같은 카테고리라도 질문의 의도(가성비/프리미엄/사용자 특성 등)에 따라 AI가 추천하는 브랜드가 달라지는가?*

## 분석 설계

- **카테고리**: 공기청정기, 로봇청소기 (고관여 + 다자 경쟁 생활가전)
- **브랜드 사전**: 카테고리별 8~9개 경쟁 브랜드와 표기 변형(제품 라인명 포함)
- **구매 의도 6유형**: 일반추천 · 가성비 · 프리미엄 · 공간상황 · 사용자특성 · 브랜드비교
- **질의 세트**: 2 카테고리 × 6 의도 × 8 변형 = 96개 질의
- **수집 조건**: 모델 내부 지식만으로 답하는 조건(`internal`, `internal_alt`)과 Google Search 연결 조건(`grounded`, 유료 쿼터 필요). LLM 응답은 확률적이라 질의×조건마다 반복 수집해 평균으로 봅니다.

### 지표

| 지표 | 정의 |
|---|---|
| SoV (노출 점유율) | 해당 브랜드가 언급된 응답 비율 |
| Top1 점유율 | 응답에서 가장 먼저 언급된 비율 (1순위 추천 자리) |
| 평균 언급 순위 | 언급됐을 때 몇 번째로 등장하는지 |

질의문이 브랜드를 직접 지목한 경우(예: "LG랑 삼성 뭐가 나아?") 해당 브랜드 언급은 유도된 언급으로 보고 SoV 집계에서 제외합니다.

## 구조

```
build_prompts.py   질의 세트 생성 → data/prompts.json
collect.py         Gemini API로 응답 수집 → data/responses.jsonl (중단 후 이어받기 지원)
analyze.py         브랜드 언급 추출 및 지표 산출 → out/*.csv
data/              질의 세트와 수집된 응답
out/               mentions.csv, sov_by_condition.csv, sov_by_intent.csv, condition_effect.csv
report.html        분석 결과 리포트
```

## 실행 방법

Python 3.10+ 와 `requests`, `pandas` 가 필요합니다.

```bash
pip install requests pandas
echo 'GEMINI_API_KEY=발급받은키' > .env

python3 build_prompts.py                 # 질의 세트 생성
python3 collect.py --list-models         # 내 키로 쓸 수 있는 모델 확인
python3 collect.py                       # 기본: internal + internal_alt, 3회 반복
python3 collect.py --status              # 수집 현황 확인
python3 analyze.py                       # 지표 산출
```

주요 옵션: `--conditions internal,grounded`, `--repeats 3`, `--limit 4`(소량 테스트), `--sleep 4.0`(호출 간 대기)

## 한계

- `grounded`(검색 연결) 조건은 무료 티어에서 쿼터가 없어 수집하지 못했습니다.
- 현재 수집된 응답은 대부분 `internal_alt` 조건입니다. 엔진 간 비교는 추가 수집이 필요합니다.
- 브랜드 언급은 정규식 기반 별칭 매칭이라 문맥(긍정/부정 언급)은 구분하지 않습니다.
