import asyncio
from src.bot.telegram_bot import build_application


def main():
    app = build_application()
    app.run_polling()


if __name__ == "__main__":
    main()
