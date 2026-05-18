import asyncio
import os
import tempfile

from telegram import Update, Document
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)

from config import TELEGRAM_BOT_TOKEN
from src.ai.refiner import map_and_refine_script, extract_emphasis, extract_slide_texts_from_pdf, calculate_cost

WAIT_INFO, WAIT_FILES = range(2)

user_sessions: dict[int, dict] = {}


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "안녕하세요! 강의 대본 페이지별 매핑 봇입니다. 📝\n\n"
        "먼저 강의 정보를 알려주세요!\n"
        "형식: 날짜_교수이름_과목명\n"
        "예시: 0413_김교수_병리학"
    )
    return WAIT_INFO


async def receive_info(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    lecture_info = update.message.text.strip()

    user_sessions[user_id] = {
        "lecture_info": lecture_info,
        "pdf_path": None,
        "txt_files": [],
    }

    await update.message.reply_text(
        f"✅ 강의 정보 저장: {lecture_info}\n\n"
        "이제 파일을 보내주세요!\n"
        "1️⃣ 강의 슬라이드 PDF\n"
        "2️⃣ 원본 강의 대본 txt 파일 (여러 개 가능)\n\n"
        "다 보내셨으면 /done 입력!"
    )
    return WAIT_FILES


async def receive_file(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if user_id not in user_sessions:
        await update.message.reply_text("먼저 /start 를 입력해주세요!")
        return ConversationHandler.END

    document = update.message.document
    fname = document.file_name.lower()

    if fname.endswith(".pdf"):
        file = await document.get_file()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            await file.download_to_drive(tmp.name)
            user_sessions[user_id]["pdf_path"] = tmp.name
        await update.message.reply_text(f"✅ PDF 수신 완료! ({document.file_name})")

    elif fname.endswith(".txt"):
        file = await document.get_file()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
            await file.download_to_drive(tmp.name)
            with open(tmp.name, "r", encoding="utf-8") as f:
                content = f.read()
        os.remove(tmp.name)
        user_sessions[user_id]["txt_files"].append({
            "name": document.file_name,
            "content": content,
        })
        txt_count = len(user_sessions[user_id]["txt_files"])
        await update.message.reply_text(f"✅ 대본 파일 {txt_count}개 수신! ({document.file_name})")

    else:
        await update.message.reply_text("PDF 또는 txt 파일만 가능합니다!")

    return WAIT_FILES


async def done(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session = user_sessions.get(user_id)

    if not session:
        await update.message.reply_text("먼저 /start 를 입력해주세요!")
        return ConversationHandler.END

    if not session["pdf_path"]:
        await update.message.reply_text("슬라이드 PDF를 먼저 보내주세요!")
        return WAIT_FILES

    if not session["txt_files"]:
        await update.message.reply_text("대본 txt 파일을 먼저 보내주세요!")
        return WAIT_FILES

    lecture_info = session["lecture_info"]
    pdf_path = session["pdf_path"]

    session["txt_files"].sort(key=lambda x: x["name"])
    raw_script = "\n\n".join([f["content"] for f in session["txt_files"]])

    status_msg = await update.message.reply_text(
        "⏳ 처리 중입니다...\n"
        "슬라이드 텍스트 추출 중..."
    )

    try:
        slide_texts = await extract_slide_texts_from_pdf(pdf_path)
        total_pages = len(slide_texts)

        await status_msg.edit_text(
            f"⏳ 처리 중입니다...\n"
            f"✅ 슬라이드 {total_pages}페이지 추출 완료\n"
            f"1/2 페이지별 매핑 + 정제 중... (0/{total_pages})"
        )

        # 페이지별 매핑 + 정제 (10장씩 진행상황 업데이트)
        results = []
        total_input = 0
        total_output = 0

        import google.generativeai as genai
        from config import GEMINI_API_KEY
        genai.configure(api_key=GEMINI_API_KEY)

        BATCH_SIZE = 10
        for batch_start in range(0, total_pages, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_pages)
            batch_texts = slide_texts[batch_start:batch_end]

            for i, slide_text in enumerate(batch_texts):
                page_num = batch_start + i + 1
                model = genai.GenerativeModel(
                    model_name="gemini-2.5-flash",
                    generation_config=genai.GenerationConfig(temperature=0.0),
                )
                prompt = f"""
당신은 의학과 강의 슬라이드와 대본을 매핑하고 정제하는 전문가입니다.

아래는 {lecture_info} 강의의 슬라이드 {page_num}페이지 내용입니다.
전체 강의 대본에서 이 슬라이드를 설명하는 부분을 찾아서 발췌하되, 동시에 아래 정제 조건에 따라 다듬어주세요.

[슬라이드 {page_num} 내용]
{slide_text}

[전체 강의 대본]
{raw_script}

[정제 조건]
1. 오탈자를 수정하되, 의학 용어에 맞게 수정 (예: 미토콘돌아 → 미토콘드리아)
2. 교수님이 학생에게 질문한 경우 대화 형식 유지 (교수님: 질문 / 학생: 답변)
3. 불필요한 미사여구 제거, 구어체는 반드시 유지할 것
4. 교수님 농담 살리기
5. 의학용어는 영어로 + 괄호 안에 한국어 번역 (예: smallpox(천연두))
6. 절대 요약하지 말 것. 산문 형식 유지
7. 강조하거나 기억하라고 한 내용 반드시 살리기
8. 관련 내용이 없으면 "해당 없음" 출력

정제된 발췌 내용만 출력하세요. 다른 설명 없이 바로 본문만 출력하세요.
"""
                response = await model.generate_content_async(prompt)
                usage = response.usage_metadata
                total_input += usage.prompt_token_count
                total_output += usage.candidates_token_count
                results.append(f"[슬라이드 {page_num}]\n{response.text.strip()}")

            await status_msg.edit_text(
                f"⏳ 처리 중입니다...\n"
                f"✅ 슬라이드 {total_pages}페이지 추출 완료\n"
                f"1/2 페이지별 매핑 + 정제 중... ({batch_end}/{total_pages})"
            )

        mapped_text = "\n\n".join(results)

        await status_msg.edit_text(
            f"⏳ 처리 중입니다...\n"
            f"✅ 1/2 페이지별 매핑 + 정제 완료\n"
            f"2/2 중요 내용 발췌 중..."
        )

        emphasis_text, emphasis_cost = await extract_emphasis(raw_script, lecture_info)
        total_input += emphasis_cost["input_tokens"]
        total_output += emphasis_cost["output_tokens"]

        total_cost = calculate_cost(total_input, total_output)

        await status_msg.edit_text("✅ 완료! 파일 전송 중...")

        await update.message.reply_document(
            document=mapped_text.encode("utf-8"),
            filename=f"{lecture_info}_페이지별정제본.txt",
            caption="📄 슬라이드 페이지별 정제된 대본입니다. 검토 후 첫 번째 봇에 사용하세요!",
        )

        await update.message.reply_document(
            document=emphasis_text.encode("utf-8"),
            filename=f"{lecture_info}_중요내용.txt",
            caption="⭐ 교수님이 강조하신 중요 내용 모음입니다.",
        )

        await update.message.reply_text(
            f"💰 이번 사용량\n"
            f"총 토큰: {total_cost['total_tokens']:,}개\n"
            f"예상 비용: 약 ₩{total_cost['cost_krw']:.1f}원"
        )

        os.remove(pdf_path)
        del user_sessions[user_id]

    except Exception as e:
        await status_msg.edit_text(f"오류가 발생했습니다: {str(e)}\n/start로 다시 시도해 주세요.")
        return ConversationHandler.END

    return ConversationHandler.END


async def cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id in user_sessions:
        session = user_sessions[user_id]
        if session.get("pdf_path") and os.path.exists(session["pdf_path"]):
            os.remove(session["pdf_path"])
        del user_sessions[user_id]
    await update.message.reply_text("취소되었습니다. /start로 다시 시작할 수 있습니다.")
    return ConversationHandler.END


def build_application() -> Application:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            WAIT_INFO: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_info)],
            WAIT_FILES: [
                MessageHandler(filters.Document.ALL, receive_file),
                CommandHandler("done", done),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    app.add_handler(conv_handler)
    return app
