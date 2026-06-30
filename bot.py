import os
import logging
import asyncio
from aiogram import Bot, Dispatcher, html
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Загружаем переменные окружения (для локального запуска, в Docker передадим напрямую)
load_dotenv()

TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Инициализация клиентов
bot = Bot(token=TOKEN)
dp = Dispatcher()

# Клиент OpenRouter API (использует совместимый с OpenAI интерфейс)
openai_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY,
)

# Хранилище контекста диалогов: {user_id: [messages_history]}
# Повышенная сложность: сохраняем контекст между запросами
USER_CONTEXT = {}
MAX_CONTEXT_LEN = 10  # Храним последние 10 реплик

@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """Обработчик команды /start"""
    user_name = message.from_user.full_name
    # Сбрасываем контекст при старте
    USER_CONTEXT[message.from_user.id] = []
    await message.answer(
        f"Привет, {html.bold(user_name)}! 👋\n"
        f"Я Telegram-бот, интегрированный с нейросетью через OpenRouter API.\n"
        f"Отправь мне любой вопрос, и я отвечу!"
    )

@dp.message(Command("help"))
async def command_help_handler(message: Message) -> None:
    """Обработчик команды /help"""
    await message.answer(
        "🤖 Команды бота:\n"
        "/start - Начать диалог и сбросить контекст\n"
        "/help - Показать это справочное сообщение\n\n"
        "Просто отправь мне текст, чтобы пообщаться с AI. Я помню контекст нашей беседы!"
    )

@dp.message()
async def chat_with_llm_handler(message: Message) -> None:
    """Обработка текстовых сообщений и отправка запроса в LLM"""
    user_id = message.from_user.id
    user_text = message.text

    # Инициализируем историю, если её нет
    if user_id not in USER_CONTEXT:
        USER_CONTEXT[user_id] = []

    # Добавляем сообщение пользователя в историю
    USER_CONTEXT[user_id].append({"role": "user", "content": user_text})

    # Держим контекст в рамках лимита
    if len(USER_CONTEXT[user_id]) > MAX_CONTEXT_LEN:
        USER_CONTEXT[user_id] = USER_CONTEXT[user_id][-MAX_CONTEXT_LEN:]

    # Отправляем статус "печатает...", пока ждем ответ от API
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")

    try:
        # Запрос к бесплатной модели через OpenRouter
        response = await openai_client.chat.completions.create(
            model="google/gemma-4-31b-it:free", 
            messages=USER_CONTEXT[user_id]
        )
        
        ai_response = response.choices[0].message.content
        
        # Добавляем ответ нейросети в историю для сохранения контекста
        USER_CONTEXT[user_id].append({"role": "assistant", "content": ai_response})
        
        # Отправляем ответ пользователю
        await message.answer(ai_response, parse_mode="Markdown")

    except Exception as e:
        logging.error(f"Ошибка при запросе к API: {e}")
        await message.answer("Произошла ошибка при обращении к нейросети. Попробуйте позже.")

async def main() -> None:
    logging.info("Запуск бота...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())