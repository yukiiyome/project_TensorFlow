import tensorflow as tf
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    BotCommand
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

logger = logging.getLogger(__name__)
# MobileNetV2
model = tf.keras.applications.MobileNetV2(weights="imagenet")


async def classify_image(image_path):
    """Классификация изображения с помощью MobileNetV2"""
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array[np.newaxis, ...])
    predictions = model.predict(img_array)
    return tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]

TOKEN = "7268432350:AAHsKSTvVRN8ZbQTKMuS9yj30iS1JAZ_g4w"

# ===== КЛАВИАТУРЫ =====
def get_main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 Анализировать другое фото", callback_data="new_analysis")],
        [InlineKeyboardButton("ℹ️ Примеры фото", callback_data="examples")],
        [InlineKeyboardButton("✍️ Оставить фидбэк", callback_data="feedback_button")]
    ])


def get_examples_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Животные", callback_data="example_животные")],
        [InlineKeyboardButton("Еда", callback_data="example_еда")],
        [InlineKeyboardButton("Достопримечательности", callback_data="example_достопримечательности")],
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
/feedback - Оставть обратную свзяь
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
        response = "🔍 *Результаты анализа изображения:*\n"
        response += "_(по версии нейросети MobileNetV2)_\n\n"

        medals = ["🥇", "🥈", "🥉"]

        for i, (_, label, prob) in enumerate(predictions[:3]):
            confidence = prob * 100
            response += f"{medals[i]} *{label}* — {confidence:.1f}%\n"

        response += "\n📌 _Чем выше процент — тем больше уверенность модели_"

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
    elif query.data == "feedback_button":
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
            "животные": ["🐶 Собака", "🐱 Кошка", "🦜 Птица"],
            "еда": ["🍎 Яблоко", "🍕 Пицца", "🍰 Торт"],
            "достопримечательности": ["🗽 Статуя Свободы", "🗼 Эйфелева башня", "🏛 Колизей"]
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

# ===== ОБРАТНАЯ СВЯЗЬ =====
async def feedback_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /feedback"""
    user_feedback = ' '.join(context.args)
    if user_feedback:
        logger.info(f"Feedback from {update.effective_user.id}: {user_feedback}")
        await update.message.reply_text(
            "✅ Спасибо! Ваш отзыв получен.",
            reply_markup=get_main_keyboard()
        )
    else:
        await update.message.reply_text(
            "❗ Пожалуйста, укажите корректный текст отзыва после команды. Пример:\n\n"
            "`/feedback Это была кошка, а не собака.`",
            parse_mode="Markdown",
            reply_markup=get_main_keyboard()
        )

# ===== НАСТРОЙКА БОТА =====
async def post_init(application):
    """Настройка команд меню"""
    await application.bot.set_my_commands([
        BotCommand("start", "Перезапустить бота"),
        BotCommand("help", "Помощь по использованию"),
        BotCommand("feedback", "Отправить отзыв")
    ])


def main():
    """Запуск бота"""
    application = Application.builder().token(TOKEN).post_init(post_init).build()

    # Регистрируем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("feedback", feedback_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    application.add_handler(CallbackQueryHandler(button_handler))

    # Запускаем бота
    application.run_polling()


if __name__ == "__main__":
    main()
