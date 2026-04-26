import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from rag_agent import RAGAssistant, RAGConfig
from telegram_bot import run_bot

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
    logger.info("Логирование настроено: уровень=%s", logging.getLevelName(log_level))


def main() -> None:
    try:
        load_environment()
        configure_logging()
        logger.info("Запуск приложения")
        if os.getenv("TELEGRAM_BOT_TOKEN", "").strip():
            logger.info("Обнаружен токен Telegram, запускаю режим бота")
            run_bot()
            return

        config = RAGConfig.from_env()
        assistant = RAGAssistant(config)
        connected = assistant.test_vector_connection("connection check")
        logger.info(
            "Проверка подключения к Pinecone: %s",
            "УСПЕХ" if connected else "ОШИБКА",
        )
        logger.info("Чтобы запустить Telegram-интерфейс, укажите TELEGRAM_BOT_TOKEN в .env.")
    except Exception as exc:
        logger.exception("Ошибка при стартовой проверке: %s", exc)


if __name__ == "__main__":
    main()
