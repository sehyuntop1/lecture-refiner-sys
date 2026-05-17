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
from src.ai.refiner import refine_script, extract_emphasis

WAIT_INFO, WAIT_FILES = range(2)

user_sessions: dict[int, dict] = {}


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "안녕하세요! 강의 대본 정제 봇입니다. 📝\n\n"
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
        "files": [],
    }

    await update.message.reply_text(
        f"✅ 강의 정보 저장: {lecture_info}\n\n"
        "이제 강의 대본 txt 파일을 보내주세요!\n"
        "여러 파일은 순서대로 보내주시면 합본으로 만들어드립니다.\n"
        "다 보내셨으면 /done 을 입력해주세요!"
    )
    return WAIT_FILES


async def receive_file(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if user_id not in user_sessions:
        await update.message.reply_text("먼저 /start 를 입력해주세요!")
        return ConversationHandler.END

    document = update.message.document

    if not document.file_name.lower().endswith(".txt"):
        await update.message.reply_text("txt 파일만 가능합니다!")
        return WAIT_FILES

    file = await update.message.document.get_file()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        await file.download_to_drive(tmp.name)
        with open(tmp.name, "r", encoding="utf-8") as f:
            content = f.read()
    os.remove(tmp.name)

    user_sessions[user_id]["files"].append({
        "name": document.file_name,
        "content": content,
    })

    file_count = len(user_sessions[user_id]["files"])
    await update.message.reply_text(
        f"✅ 파일 {file_count}개 수신 완료! ({document.file_name})\n"
        "더 보내시려면 계속 보내주세요. 완료되면 /done 입력!"
    )
    return WAIT_FILES


async def done(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if user_id not in user_sessions or not user_sessions[user_id]["files"]:
        await update.message.reply_text("파일을 먼저 보내주세요!")
        return WAIT_FILES

    session = user_sessions[user_id]
    lecture_info = session["lecture_info"]
    files = session["files"]

    # 파일명 순서대로 정렬 후 합본
    files.sort(key=lambda x: x["name"])
    raw_script = "\n\n".join([f["content"] for f in files])

    status_msg = await update.message.reply_text(
        "⏳ 처리 중입니다...\n"
        f"총 {len(files)}개 파일 합본 처리 중...\n"
        "1/2 대본 정제 중 (Gemini temperature=0)..."
    )

    try:
        # 1단계: 정제
        refined = await refine_script(raw_script, lecture_info)
        await status_msg.edit_text(
            "⏳ 처리 중입니다...\n"
            "✅ 1/2 대본 정제 완료\n"
            "2/2 중요 내용 발췌 중..."
        )

        # 2단계: 중요 내용 발췌
        emphasis = await extract_emphasis(raw_script, lecture_info)
        await status_msg.edit_text("✅ 완료! 파일 전송 중...")

        # 정제본 txt 전송
        refined_bytes = refined.encode("utf-8")
        await update.message.reply_document(
            document=refined_bytes,
            filename=f"{lecture_info}_정제본.txt",
            caption="📝 정제된 강의 대본입니다.",
        )

        # 중요 내용 txt 전송
        emphasis_bytes = emphasis.encode("utf-8")
        await update.message.reply_document(
            document=emphasis_bytes,
            filename=f"{lecture_info}_중요내용.txt",
            caption="⭐ 교수님이 강조하신 중요 내용 모음입니다.",
        )

        del user_sessions[user_id]

    except Exception as e:
        await status_msg.edit_text(f"오류가 발생했습니다: {str(e)}\n/start로 다시 시도해 주세요.")
        return ConversationHandler.END

    return ConversationHandler.END


async def cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id in user_sessions:
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
                MessageHandler(filters.Document.TXT, receive_file),
                CommandHandler("done", done),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    app.add_handler(conv_handler)
    return app
