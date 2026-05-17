from src.ai.gemini_client import generate

REFINE_PROMPT = """
당신은 의학과 강의 대본을 다듬는 전문가입니다.
아래는 {lecture_info} 강의 대본입니다. 다음 조건을 반드시 지켜 다듬어 주세요.

[조건1] 오탈자를 수정하되, 의학과에서 쓰는 용어에 맞게 오탈자 수정
- 예시) 미토콘돌아 → 미토콘드리아

[조건2] 교수님이 예시로 들기 위해 학생에게 질문을 했다면 대화 형식을 지켜 살릴 것
- 예시) 교수님: 질문내용 / 학생: 답변내용

[조건3] 불필요한 미사여구는 지우고 맥락에 맞게 문장을 읽기 쉬운 형태로 바꿀 것.
단, 문장의 형식을 지키기 위해 언급하셨던 의학 관련 단어들은 반드시 살려야 함
- 예시) cell의 aging은... 어 노화가 이제 → cell의 aging은 이제

[조건4] 교수님의 농담을 지우지 않고 반드시 살리기

[조건5] 의학용어가 쓰여있다면 영어로 바꿔주고 옆에 괄호로 한국어 번역을 해줄 것
- 예시) smallpox(천연두)

[조건6] 내용을 정리, 요약하여 줄이지 말고 교수님의 강의라는 틀을 벗어나지 말 것.
구두로 말했다는 걸 기억하고 산문 형식을 엄수

[조건7] 예시로 든 내용도 담고 절대로 텍스트의 내용을 임의로 요약하지 말 것. 절대로 요약하지 말 것!

[조건8] 교수님이 기억하라고 하거나, 상식으로 알아두라거나 하는 언급이 있는 건 반드시 내용을 살려두기

[조건9] 파일이 여러 개라면 순번 순서대로 이어붙여 하나의 합본으로 만들 것

원본 대본:
{raw_script}

다듬은 대본만 출력하세요. 다른 설명이나 머릿말 없이 바로 본문만 출력하세요.
"""

EMPHASIS_PROMPT = """
당신은 의학과 강의 대본에서 중요한 내용을 발췌하는 전문가입니다.
아래는 {lecture_info} 강의 대본입니다.

교수님이 강조한 내용을 모두 발췌해 주세요.
강조 표현의 예시: "중요합니다", "외우세요", "기억하세요", "시험에 나옵니다", "꼭 알아두세요",
"반드시", "알고 계셔야", "핵심은", "포인트는" 등
키워드에 국한하지 말고 문맥을 보고 교수님이 강조한 내용이라고 판단되면 모두 포함하세요.

각 항목은 아래 형식으로 작성하세요:
[중요] 발췌한 내용

원본 대본:
{raw_script}

중요 내용 발췌본만 출력하세요. 다른 설명 없이 바로 발췌 내용만 출력하세요.
"""


async def refine_script(raw_script: str, lecture_info: str) -> str:
    prompt = REFINE_PROMPT.format(
        lecture_info=lecture_info,
        raw_script=raw_script,
    )
    return await generate(prompt, temperature=0.0)


async def extract_emphasis(raw_script: str, lecture_info: str) -> str:
    prompt = EMPHASIS_PROMPT.format(
        lecture_info=lecture_info,
        raw_script=raw_script,
    )
    return await generate(prompt, temperature=0.0)
