from __future__ import annotations

import logging
import os
from pathlib import Path

import telebot
from dotenv import load_dotenv

from rag_agent import RAGAssistant, RAGConfig


MAX_TELEGRAM_MESSAGE_LEN = 3800
logger = logging.getLogger(__name__)


def load_environment() -> None:
    env_path = Path(__file__).with_name(".env")
    load_dotenv(dotenv_path=env_path, override=False)


def configure_logging() -> None:
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger.info(
        "Логирование Telegram-бота настроено: уровень=%s",
        logging.getLevelName(log_level),
    )


def split_for_telegram(text: str, limit: int = MAX_TELEGRAM_MESSAGE_LEN) -> list[str]:
    compact = text.strip()
    if not compact:
        return ["Пустой ответ от ассистента."]
    return [compact[i : i + limit] for i in range(0, len(compact), limit)]


def build_bot() -> telebot.TeleBot:
    load_environment()
    config = RAGConfig.from_env()
    assistant = RAGAssistant(config)
    logger.info("Telegram-бот инициализирован с RAG-ассистентом")

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not bot_token:
        raise ValueError("Missing required environment variable: TELEGRAM_BOT_TOKEN")

    bot = telebot.TeleBot(bot_token, parse_mode=None)

    @bot.message_handler(commands=["start", "help"])
    def send_help(message: telebot.types.Message) -> None:
        logger.info("Получена команда /start или /help от chat_id=%s", message.chat.id)
        help_text = (
            "Я ваш RAG-ассистент.\n\n"
            "Что я умею:\n"
            "- отвечать на вопросы, используя векторный поиск перед ответом;\n"
            "- запоминать важные факты о вас (эвристики + эмбеддинги);\n"
            "- загружать HTML-страницы в базу знаний.\n\n"
            "Команды:\n"
            "/add_url <https://example.com/page> - добавить URL в Pinecone\n"
            "/status - проверить подключение к векторной БД"
        )
        bot.reply_to(message, help_text)

    @bot.message_handler(commands=["status"])
    def check_status(message: telebot.types.Message) -> None:
        logger.info("Получена команда /status от chat_id=%s", message.chat.id)
        ok = assistant.test_vector_connection("telegram status check")
        status_text = "Статус векторной БД: OK" if ok else "Статус векторной БД: ОШИБКА"
        bot.reply_to(message, status_text)

    @bot.message_handler(commands=["add_url"])
    def add_url(message: telebot.types.Message) -> None:
        logger.info("Получена команда /add_url от chat_id=%s", message.chat.id)
        payload = message.text or ""
        parts = payload.split(maxsplit=1)
        if len(parts) < 2:
            bot.reply_to(message, "Использование: /add_url https://example.com/page")
            return
        url = parts[1].strip()
        if not url:
            bot.reply_to(message, "Пожалуйста, отправьте корректный URL после /add_url.")
            return

        bot.send_chat_action(message.chat.id, "typing")
        try:
            added = assistant.ingest_url(url)
            logger.info("URL проиндексирован из Telegram: url=%s chunks=%d", url, added)
            bot.reply_to(message, f"URL проиндексирован. Добавлено чанков: {added}")
        except Exception as exc:
            logger.exception("Не удалось проиндексировать URL из Telegram: %s", url)
            bot.reply_to(message, f"Не удалось проиндексировать URL: {exc}")

    @bot.message_handler(content_types=["text"])
    def chat(message: telebot.types.Message) -> None:
        text = message.text or ""
        if not text.strip():
            bot.reply_to(message, "Отправьте текстовое сообщение.")
            return

        user = message.from_user
        user_id = str(user.id) if user else None
        user_name = None
        if user:
            user_name = user.username or " ".join(
                part for part in [user.first_name, user.last_name] if part
            )
            user_name = user_name.strip() or None

        bot.send_chat_action(message.chat.id, "typing")
        logger.info(
            "Получено текстовое сообщение: chat_id=%s user_id=%s text_len=%d",
            message.chat.id,
            user_id,
            len(text),
        )
        try:
            answer = assistant.handle_user_message(
                user_message=text,
                user_id=user_id,
                user_name=user_name,
            )
        except Exception as exc:
            logger.exception("Ошибка при обработке сообщения пользователя")
            bot.reply_to(message, f"Внутренняя ошибка: {exc}")
            return

        response_chunks = split_for_telegram(answer)
        logger.info("Отправка частей ответа: count=%d", len(response_chunks))
        for chunk in response_chunks:
            bot.send_message(message.chat.id, chunk)

    return bot


def run_bot() -> None:
    configure_logging()
    logger.info("Запуск цикла polling для Telegram-бота")
    bot = build_bot()
    bot.infinity_polling(skip_pending=True, timeout=60, long_polling_timeout=60)


if __name__ == "__main__":
    run_bot()
