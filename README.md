# VPg04: RAG-ассистент с Telegram-ботом

Легковесный Python-проект с RAG-логикой: ассистент ищет контекст в Pinecone, учитывает персональную память пользователя и отвечает через Telegram.

## Что умеет

- индексирует веб-страницы по URL в векторную базу;
- делает retrieval по базе знаний и по персональной памяти;
- сохраняет важные факты о пользователе без дублей;
- отвечает на русском/английском в зависимости от языка запроса;
- подтягивает актуальную погоду из Open-Meteo при погодных вопросах;
- добавляет в ответ блок `Источники`.

## Содержание

- [Структура проекта](#структура-проекта)
- [Как работает система](#как-работает-система)
- [Быстрый старт](#быстрый-старт)
- [Команды Telegram](#команды-telegram)
- [Данные в Pinecone](#данные-в-pinecone)
- [Проверка работоспособности](#проверка-работоспособности)
- [Тюнинг качества](#тюнинг-качества)
- [Безопасность](#безопасность)

## Структура проекта

- `main.py` - точка входа:
  - загружает `.env`;
  - настраивает логирование;
  - запускает Telegram-бота, если есть `TELEGRAM_BOT_TOKEN`;
  - иначе выполняет smoke-test Pinecone.
- `telegram_bot.py` - Telegram-интерфейс (polling), команды, разбиение длинных ответов.
- `rag_agent.py` - вся RAG-логика:
  - `RAGConfig` и валидация переменных окружения;
  - ingest URL -> split -> embeddings -> upsert в Pinecone;
  - retrieval по KB и user-memory;
  - эвристики выделения персональных фактов;
  - интеграция Open-Meteo для runtime API-контекста;
  - генерация и постобработка ответа.
- `env.example` - шаблон `.env`.
- `requirements.txt` - зависимости.
- `example.py` - отдельный демонстрационный пример (в основном запуске не используется).

## Как работает система

### 1) Старт приложения

1. Загружается `.env` через `load_dotenv`.
2. Поднимается логирование (`LOG_LEVEL`, по умолчанию `INFO`).
3. Создается `RAGConfig.from_env()`.
4. Проверяются обязательные переменные:
   - `OPENAI_API_KEY`
   - `OPENAI_BASE_URL`
   - `PINECONE_API_KEY`
   - `PINECONE_INDEX_NAME`
5. Инициализируются:
   - `ChatOpenAI`
   - `OpenAIEmbeddings`
   - Pinecone index
   - `RecursiveCharacterTextSplitter`

### 2) Режим запуска

- есть `TELEGRAM_BOT_TOKEN` -> запускается `run_bot()`;
- нет `TELEGRAM_BOT_TOKEN` -> выполняется `test_vector_connection()`.

### 3) Обработка сообщения пользователя

1. **Извлечение URL**  
   Из текста выделяются ссылки регулярным выражением.

2. **Индексация URL**  
   Для каждой ссылки:
   - загружается HTML (`urllib`);
   - удаляются `script/style/noscript`;
   - HTML преобразуется в текст (`BeautifulSoup`);
   - текст делится на чанки;
   - строятся эмбеддинги;
   - чанки сохраняются в Pinecone (`type=kb_chunk`, `source=url`).

3. **Сохранение персональной памяти**  
   Если есть `user_id`, ассистент:
   - выделяет кандидаты на персональные факты;
   - оценивает важность;
   - отсекает слабые и дублирующиеся факты;
   - сохраняет новые записи в Pinecone (`type=user_memory`).

4. **Runtime API-контекст для погоды**  
   Если запрос о погоде/температуре/ветре:
   - вызывается Open-Meteo;
   - по возможности определяется регион по тексту;
   - если регион не найден, используется Москва по умолчанию.

5. **Retrieval + генерация ответа**  
   - ищется релевантный KB-контекст;
   - отдельно подтягивается память пользователя;
   - формируется prompt из 3 блоков:
     - `Knowledge Base Context`
     - `Personal Memory`
     - `Runtime API Context`
   - LLM генерирует ответ на языке пользователя.

6. **Постобработка**  
   - исправляется ложная фраза "нет доступа к API", если runtime API уже получен;
   - в конец добавляется блок `Источники`;
   - при ошибках ingest/API добавляются предупреждения.

## Быстрый старт

### 1) Установка

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Настройка `.env`

Скопируйте `env.example` в `.env` и заполните значения.

Обязательные:
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`

Рекомендуемые:
- `OPENAI_CHAT_MODEL`
- `OPENAI_EMBEDDING_MODEL`
- `PINECONE_NAMESPACE`
- `LOG_LEVEL`

Для Telegram-режима:
- `TELEGRAM_BOT_TOKEN`

Для тюнинга RAG:
- `RAG_CHUNK_SIZE`
- `RAG_CHUNK_OVERLAP`
- `RAG_TOP_K`
- `RAG_KB_MIN_SCORE`
- `RAG_KB_MIN_SCORE_WEATHER`
- `USER_MEMORY_TOP_K`
- `USER_MEMORY_MIN_IMPORTANCE`
- `USER_MEMORY_DEDUPE_THRESHOLD`

### 3) Запуск

```powershell
python main.py
```

Поведение:
- с `TELEGRAM_BOT_TOKEN` -> запускается бот;
- без `TELEGRAM_BOT_TOKEN` -> только проверка Pinecone.

## Команды Telegram

- `/start` и `/help` - краткая справка;
- `/status` - проверка доступности векторной БД;
- `/add_url <url>` - ручная индексация страницы;
- любое текстовое сообщение - ответ ассистента с RAG-контекстом.

## Данные в Pinecone

В рамках `PINECONE_NAMESPACE` используются типы:
- `kb_chunk` - чанки базы знаний;
- `user_memory` - персональные факты пользователя;
- `integration_test` - временные записи smoke-test (сразу удаляются).

## Проверка работоспособности

1. Запустите `python main.py` без Telegram-токена.
2. Убедитесь, что smoke-test Pinecone проходит.
3. Добавьте `TELEGRAM_BOT_TOKEN` и перезапустите приложение.
4. В Telegram выполните:
   - `/status`
   - `/add_url https://example.com`
   - вопрос по проиндексированной странице
5. Проверьте, что в ответе есть блок `Источники`.

## Тюнинг качества

- `RAG_TOP_K`: меньше -> точнее и уже контекст; больше -> шире охват.
- `RAG_KB_MIN_SCORE`: фильтр "шумных" KB-совпадений.
- `RAG_KB_MIN_SCORE_WEATHER`: более строгий порог для погодных сценариев.
- `USER_MEMORY_DEDUPE_THRESHOLD`: чувствительность к дублям памяти.
- `USER_MEMORY_MIN_IMPORTANCE`: минимальная значимость факта для сохранения.

## Безопасность

- не коммитьте `.env` и ключи API;
- ограничивайте доступ к Pinecone и Telegram token;
- проверяйте источники, которые индексируете через URL;
- для production добавьте отдельные namespace/индексы под среды.

