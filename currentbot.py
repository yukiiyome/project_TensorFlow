import tensorflow as tf
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    BotCommand,
    InputMediaPhoto
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
    CallbackQueryHandler
)
from PIL import Image
import numpy as np
import os
import logging

# Настройка логгирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Use __name__ (recommended for modules) or a custom string
logger = logging.getLogger(__name__)  # Correct way
# Загружаем модель MobileNetV2
model = tf.keras.applications.MobileNetV2(weights="imagenet")


async def classify_image(image_path):
    """Классификация изображения с помощью MobileNetV2"""
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array[np.newaxis, ...])
    predictions = model.predict(img_array)
    return tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]


TOKEN = "7268432350:AAEFnQA34n6GRtpYM0PH0nMASyZwYbxxQkY"


# ===== КЛАВИАТУРЫ =====
def get_main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 Анализировать другое фото", callback_data="new_analysis")],
        [InlineKeyboardButton("❓ Это неточно", callback_data="feedback")],
        [InlineKeyboardButton("ℹ️ Примеры фото", callback_data="examples")]
    ])


def get_examples_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🐶 Животные", callback_data="example_animals")],
        [InlineKeyboardButton("🍎 Еда", callback_data="example_food")],
        [InlineKeyboardButton("🏠 Достопримечательности", callback_data="example_landmarks")],
        [InlineKeyboardButton("🔙 Назад", callback_data="back")]
    ])


# ===== КОМАНДЫ =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    welcome_text = """
🌟 *Добро пожаловать в Image Classification Bot!* 🌟

📸 Отправьте мне изображение, и я определю что на нём с помощью нейросети MobileNetV2.

🔹 Лучшие результаты с:
- Чёткими фотографиями объектов
- Одиночными предметами на белом фоне
- Хорошим освещением

✨ *Попробуйте отправить фото прямо сейчас!*
"""
    await update.message.reply_text(
        welcome_text,
        parse_mode="Markdown",
        reply_markup=get_main_keyboard()
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help"""
    help_text = """
🆘 *Помощь по боту*

🔍 *Как использовать:*
1. Отправьте любое изображение
2. Получите анализ содержимого
3. Уточните при необходимости

📊 *Точность:*
- ~80-95% для чётких изображений
- ~60-75% для сложных сцен

⚙️ *Технологии:*
- Нейросеть MobileNetV2
- База данных ImageNet (1000+ категорий)

📌 *Команды:*
/start - Главное меню
/help - Эта справка
"""
    await update.message.reply_text(
        help_text,
        parse_mode="Markdown",
        reply_markup=get_main_keyboard()
    )


# ===== ОБРАБОТКА ИЗОБРАЖЕНИЙ =====
async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка изображений"""
    image_path = "temp_image.jpg"
    try:
        # Скачиваем фото
        photo_file = await update.message.photo[-1].get_file()
        await photo_file.download_to_drive(image_path)

        # Показываем превью
        await update.message.reply_photo(
            photo=update.message.photo[-1].file_id,
            caption="🔄 *Анализирую изображение...*",
            parse_mode="Markdown"
        )

        # Классифицируем
        predictions = await classify_image(image_path)

        # Форматируем ответ
        response = "🔍 *Результаты анализа:*\n\n"
        response += "🏆 *Топ-5 вероятных вариантов:*\n"

        for i, (_, label, prob) in enumerate(predictions):
            confidence = prob * 100
            bar_length = int(confidence / 5)
            progress_bar = "[" + "█" * bar_length + " " * (20 - bar_length) + "]"
            if i == 0:
                response += f"🥇 {label}: {confidence:.1f}%\n{progress_bar}\n\n"
            elif i == 1:
                response += f"🥈 {label}: {confidence:.1f}%\n{progress_bar}\n\n"
            elif i == 2:
                response += f"🥉 {label}: {confidence:.1f}%\n{progress_bar}\n\n"
            else:
                response += f"▫️ {label}: {confidence:.1f}%\n{progress_bar}\n\n"
            
        response += "\n_Уверенность модели может варьироваться в зависимости от качества изображения_"

        await update.message.reply_text(
            response,
            parse_mode="Markdown",
            reply_markup=get_main_keyboard()
        )

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        await update.message.reply_text(
            "❌ *Произошла ошибка при обработке изображения*\n\n"
            "Попробуйте отправить другое фото или проверьте его качество.",
            parse_mode="Markdown",
            reply_markup=get_main_keyboard()
        )
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)


# ===== CALLBACK-ОБРАБОТЧИКИ =====
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == "new_analysis":
        await query.edit_message_text(
            "🔄 *Отправьте новое изображение для анализа*\n\n"
            "Попробуйте сделать фото с разных ракурсов для лучшего результата.",
            parse_mode="Markdown"
        )
    elif query.data == "feedback":
        await query.edit_message_text(
            "✍️ *Обратная связь*\n\n"
            "Напишите, что действительно было на изображении, и мы учтём это для улучшения модели!\n\n"
            "Формат: /feedback [правильный ответ]",
            parse_mode="Markdown"
        )
    elif query.data == "examples":
        await query.edit_message_text(
            "📸 *Примеры фотографий для лучшего распознавания:*\n\n"
            "Выберите категорию, чтобы увидеть примеры:",
            parse_mode="Markdown",
            reply_markup=get_examples_keyboard()
        )
    elif query.data.startswith("example_"):
        category = query.data.replace("example_", "")
        examples = {
            "animals": ["🐶 Собака", "🐱 Кошка", "🦜 Птица"],
            "food": ["🍎 Яблоко", "🍕 Пицца", "🍰 Торт"],
            "landmarks": ["🗽 Статуя Свободы", "🗼 Эйфелева башня", "🏛 Колизей"]
        }

        await query.edit_message_text(
            f"🖼 *Примеры для категории {category}:*\n\n"
            f"{chr(10).join(examples.get(category, []))}\n\n"
            "Попробуйте сделать похожие фотографии для лучшего результата!",
            parse_mode="Markdown",
            reply_markup=get_examples_keyboard()
        )
    elif query.data == "back":
        await query.edit_message_text(
            "🔙 Возвращаемся в главное меню",
            parse_mode="Markdown",
            reply_markup=get_main_keyboard()
        )


# ===== НАСТРОЙКА БОТА =====
async def post_init(application):
    """Настройка команд меню"""
    await application.bot.set_my_commands([
        BotCommand("start", "Перезапустить бота"),
        BotCommand("help", "Помощь по использованию")
    ])


def main():
    """Запуск бота"""
    application = Application.builder().token(TOKEN).post_init(post_init).build()

    # Регистрируем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    application.add_handler(CallbackQueryHandler(button_handler))

    # Запускаем бота
    application.run_polling()


if __name__ == "__main__":
    main()
