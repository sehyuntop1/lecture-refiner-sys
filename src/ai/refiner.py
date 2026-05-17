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
당신은 의학과 시험 준비를 돕는 전문가입니다.
아래는 {lecture_info} 강의 대본입니다.

다음 기준에 따라 시험에 출제될 가능성이 높은 의학적 내용만 엄격하게 발췌해 주세요.

[발췌 기준]
- 교수님이 "시험에 나온다", "외워라", "반드시 알아야 한다", "중요하다", "기억해라" 등 명시적으로 강조한 의학 내용
- 교수님이 특정 수치, 메커니즘, 질환명, 용어를 반복해서 강조한 내용
- 교수님이 "이것만큼은", "핵심은", "포인트는" 등으로 콕 집어 언급한 내용

[절대 포함하지 말 것]
- AI, 기술 사용법, 수업 운영 방식에 대한 당부
- 일상적인 공지사항, 출석, 과제 관련 내용
- 단순한 예시나 비유로 든 내용 (강조 없이)
- 교수님 농담이나 잡담

각 항목은 아래 형식으로 작성하세요:
[중요] 발췌한 의학 내용

원본 대본:
{raw_script}

중요 의학 내용 발췌본만 출력하세요. 다른 설명 없이 바로 발췌 내용만 출력하세요.
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
