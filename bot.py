import os
import re
import json
import asyncio
from collections import defaultdict
from PIL import Image
from io import BytesIO

from dotenv import load_dotenv
from bytez import Bytez
from groq import Groq
from telegram import Update
from telegram.ext import (
    Application,
    MessageHandler,
    ContextTypes,
    filters
)

# ================= LOAD ENV =================
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
BYTEZ_KEY = os.getenv("BYTEZ_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")

if not BOT_TOKEN or not BYTEZ_KEY or not GROQ_KEY:
    raise RuntimeError("Missing environment variables")

# ================= CLIENTS =================
bytez = Bytez(BYTEZ_KEY)
ocr_model = bytez.model("microsoft/trocr-large-printed")

groq = Groq(api_key=GROQ_KEY)
TEXT_MODEL = "llama-3.1-8b-instant"

# ================= REGEX =================
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")

# ================= ALBUM BUFFER =================
album_buffer = defaultdict(list)

# ================= OCR =================
def run_ocr(image_bytes: bytes) -> str:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    output, error = ocr_model.run(img)
    if error:
        return ""
    return output.strip()

# ================= STRUCTURE =================
def structure_text(text: str) -> dict:
    prompt = f"""
Extract LinkedIn profile details.

Rules:
- Use ONLY provided text
- If missing, write "N/A"
- Do NOT guess
- Output valid JSON only

Fields:
name
profession
current_company
current_location
email
phone

TEXT:
{text}
"""
    res = groq.chat.completions.create(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    content = res.choices[0].message.content.strip()

    try:
        data = json.loads(content)
    except:
        data = {}

    return data

# ================= REGEX RESCUE =================
def regex_rescue(data: dict, text: str) -> dict:
    if data.get("email", "N/A") == "N/A":
        m = EMAIL_RE.search(text)
        if m:
            data["email"] = m.group(0)

    if data.get("phone", "N/A") == "N/A":
        m = PHONE_RE.search(text)
        if m:
            data["phone"] = m.group(0)

    for k in ["name","profession","current_company","current_location","email","phone"]:
        if not data.get(k):
            data[k] = "N/A"

    return data

# ================= FORMAT =================
def format_reply(d: dict) -> str:
    return (
        "📌 **Extracted Profile**\n\n"
        f"👤 Name: {d['name']}\n"
        f"💼 Profession: {d['profession']}\n"
        f"🏢 Company: {d['current_company']}\n"
        f"📍 Location: {d['current_location']}\n"
        f"📧 Email: {d['email']}\n"
        f"📞 Phone: {d['phone']}"
    )

# ================= HANDLER =================
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    chat_id = msg.chat_id
    group_id = msg.media_group_id or msg.message_id

    photo = msg.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    image_bytes = await file.download_as_bytearray()

    album_buffer[group_id].append((msg, image_bytes))

    await asyncio.sleep(1.5)

    if len(album_buffer[group_id]) == 0:
        return

    entries = album_buffer.pop(group_id)
    merged_text = ""

    for _, img_bytes in entries:
        merged_text += "\n" + run_ocr(img_bytes)

    structured = structure_text(merged_text)
    final = regex_rescue(structured, merged_text)

    reply_to = entries[0][0]
    await reply_to.reply_text(
        format_reply(final),
        reply_to_message_id=reply_to.message_id,
        parse_mode="Markdown"
    )

# ================= MAIN =================
def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    print("🤖 Bot running...")
    app.run_polling()

if __name__ == "__main__":
    main()
