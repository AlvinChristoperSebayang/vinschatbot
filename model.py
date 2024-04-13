from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from langdetect import detect

# Inisialisasi
stop_words_en = set(stopwords.words('english'))
stop_words_id = set(stopwords.words('indonesian'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text, language):
    stop_words = stop_words_en if language == 'en' else stop_words_id
    tokens = word_tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return filtered_tokens

# Daftar pertanyaan
questions = [
    "when the store is open?",
    "dimana toko anda berada?"
]

# Kamus jawaban
answers = {
    "schedule": {
        "en": "The store opens at 8 AM and closes at 9 PM every day.",
        "id": "Toko buka pada pukul 8 pagi dan tutup pukul 9 malam setiap hari."
    },
    "promo": {
        "en": "There are no discounts currently. Please visit our website for the latest promotions.",
        "id": "Tidak ada diskon saat ini. Silakan kunjungi website kami untuk promo terbaru."
    },
    "location": {
        "en": "Our store is located at 123 Main Street, City.",
        "id": "Toko kami berlokasi di Jalan Utama No. 123, Kota."
    }
}

# Mengolah pertanyaan
for question in questions:
    language = detect(question)
    preprocessed_question = preprocess_text(question, language)
    print("Pertanyaan ({lang}): {question}".format(lang=language, question=question))
    
    # Mendeteksi topik
    # Di sini Anda dapat menggunakan model klasifikasi untuk mendeteksi topik berdasarkan pertanyaan yang diproses
    # Sebagai contoh sederhana, kita hanya menggunakan kata kunci
    if any(word in preprocessed_question for word in ["open", "buka"]):
        topic = "schedule"
    elif any(word in preprocessed_question for word in ["discount", "diskon"]):
        topic = "promo"
    elif any(word in preprocessed_question for word in ["location", "lokasi"]):
        topic = "location"
    else:
        topic = None
    
    if topic:
        # Mengambil jawaban berdasarkan topik dan bahasa
        answer = answers[topic].get(language, "Maaf, jawaban tidak tersedia untuk bahasa ini.")
        print("Jawaban: ", answer)
    else:
        print("Topik tidak dikenali, mohon maaf kami tidak dapat memberikan informasi yang Anda cari.")
