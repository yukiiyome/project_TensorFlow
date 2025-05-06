import tensorflow as tf
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from PIL import Image
import numpy as np
import os

# Загружаем модель MobileNetV2
model = tf.keras.applications.MobileNetV2(weights="imagenet")

async def classify_image(image_path):
    """Классификация изображения с помощью MobileNetV2"""
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array[np.newaxis, ...])
    predictions = model.predict(img_array)
    return tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]

TOKEN = "7268432350:AAEFnQA34n6GRtpYM0PH0nMASyZwYbxxQkY"

# Создаем клавиатуру с кнопками
reply_keyboard = [
    [KeyboardButton("/help"), KeyboardButton("/start")]
]
markup = ReplyKeyboardMarkup(reply_keyboard, resize_keyboard=True, input_field_placeholder="Выберите команду...")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    welcome_text = """
📷 *Image Classification Bot*
Отправьте мне изображение, и я определю что на нем!
    
Доступные команды:
/start - Перезапустить бота
/help - Получить справку
"""
    await update.message.reply_text(welcome_text, parse_mode="Markdown", reply_markup=markup)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help"""
    help_text = """
🆘 *Помощь по боту*
    
Этот бот использует нейросеть MobileNetV2 для классификации изображений. Просто отправьте любое изображение, и бот попытается определить что на нем.

🔹 Поддерживаются: фото животных, растений, техники и многое другое
🔹 Точность: ~75-90% в зависимости от предмета
🔹 Лучшие результаты с четкими фотографиями объектов
    
Команды:
/start - Главное меню
/help - Эта справка
"""
    await update.message.reply_text(help_text, parse_mode="Markdown", reply_markup=markup)

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка изображений"""
    try:
        # Скачиваем фото
        photo_file = await update.message.photo[-1].get_file()
        image_path = "temp_image.jpg"
        await photo_file.download_to_drive(image_path)
        
        # Классифицируем
        predictions = await classify_image(image_path)
        
        # Форматируем ответ
        response = "🔍 *Результаты анализа*:\n"
        response += "\n".join([f"▫️ {label} ({prob*100:.2f}%)" for _, label, prob in predictions])
        
        await update.message.reply_text(response, parse_mode="Markdown", reply_markup=markup)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {str(e)}", reply_markup=markup)
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

def main():
    """Запуск бота"""
    application = Application.builder().token(TOKEN).build()
    
    # Регистрируем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    
    # Запускаем бота
    application.run_polling()

if __name__ == "__main__":
    main()
