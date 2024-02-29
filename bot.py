import telebot
from telebot import apihelper

from chat import Chat
from rate_limiter import RateLimiter

import dotenv
dotenv.load_dotenv()
import os

chat = Chat()
limiter = RateLimiter()
bot = telebot.TeleBot(os.getenv("BOT_TOKEN"))

@bot.message_handler(commands=["start"], chat_types=['private'])
def start(message):
    bot.reply_to(message, """سلام عزیز دل
من کیانوشم. یه چند روز واسه پشتیبانی رویداد قراره اینجا جواب سوالاتتونو بدم.
اگه سوالی برات پیش اومد دستور /ask رو بزن""")

@bot.message_handler(commands=["ask"], chat_types=['private'])
def ask(message):
    bot.reply_to(message, f"""جانم {message.from_user.first_name}
چه سوالی ازم داری؟""")
    chat.add_user(message.from_user.id)
    
@bot.message_handler(content_types=['text'], chat_types=['private'])
def get_response(message):
    if not chat.has_user(message.from_user.id):
        bot.reply_to(message, "اگه سوالی ازم داری اول دستور /ask رو وارد کن.")
        return
    if not limiter.access(message.from_user.id):
        bot.reply_to(message, """چند دقیقه دیگه سوال بپرس.
سوالات زیاد بود یکمی گیج شدم :(((((""")
        return
    if chat.count_messages(message.from_user.id) > 5:
        bot.reply_to(message, """یادم رفت داشتیم درباره چی حرف می‌زدیم.
میشه دوباره سوالتو با /ask از اول بپرسی؟""")
        return
    response = chat.chat(message.from_user.id, message.text)
    bot.reply_to(message, response)

@bot.message_handler(
    content_types=[
        'audio', 'document', 'animation', 'photo', 'sticker', 'video', 'voice'
    ],
    chat_types=['private'],
)
def unsupported_content(message):
    bot.reply_to(message, """میشه تکست بدی فقط؟ اینجا نت 3G عه :((((((""")
    
@bot.message_handler(
    content_types=[
        'text', 'audio', 'document', 'animation', 'photo', 'sticker', 'video', 'voice'
    ],
    chat_types=['group', 'supergroup', 'channel'],
)
def unsupported_chat(message):
    bot.reply_to(message, """راستش من یکمی خجالتیم.
میشه حرفامونو تو پی‌وی بزنیم؟ :)))))""")

bot.polling()