import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.classify import NaiveBayesClassifier
import random
from langdetect import detect

# Kamus jawaban
answers = {
    "schedule": {
        "en": [
            "The store opens at 9 AM and closes at 6 PM on weekdays. We are open until 10 PM on Fridays and Saturdays.",
            "Our store is open from 9 AM to 6 PM on weekdays and until 10 PM on Fridays and Saturdays.",
            "The store's hours are 9 AM to 6 PM on weekdays and extended to 10 PM on Fridays and Saturdays."
        ],
        "id": [
            "Toko buka pada pukul 9 pagi dan tutup pukul 6 sore di hari kerja. Kami buka hingga pukul 10 malam pada Jumat dan Sabtu.",
            "Jam buka toko kami adalah dari jam 9 pagi hingga 6 sore pada hari kerja dan diperpanjang hingga jam 10 malam pada Jumat dan Sabtu.",
            "Jam operasional toko kami adalah dari jam 9 pagi hingga 6 sore pada hari kerja dan diperpanjang hingga jam 10 malam pada Jumat dan Sabtu."
        ]
    },
    "promo": {
        "en": [
            "We currently have a promotion offering 20% off all items in the [product category]. The promotion ends on [end date].",
            "Get 20% off on all items in the [product category] with our current promotion. Hurry, ends on [end date]!",
            "Don't miss our latest promotion! Enjoy 20% off on all items in the [product category] until [end date]."
        ],
        "id": [
            "Saat ini kami memiliki promo yang menawarkan diskon 20% untuk semua produk dalam kategori [kategori produk]. Promo berakhir pada [tanggal berakhir].",
            "Dapatkan diskon 20% untuk semua produk dalam kategori [kategori produk] dengan promo kami saat ini. Buruan, berakhir pada [tanggal berakhir]!",
            "Jangan lewatkan promo terbaru kami! Nikmati diskon 20% untuk semua produk dalam kategori [kategori produk] hingga [tanggal berakhir]."
        ]
    },
    "location": {
        "en": [
            "Our closest store is located in City A. For more information on our other store locations, please visit our website.",
            "You can find our nearest store in City A. Visit our website for details on our other locations.",
            "The closest store to you is in City A. Check out our website to find more locations near you."
        ],
        "id": [
            "Toko terdekat kami berlokasi di Kota A. Untuk informasi lebih lanjut mengenai lokasi toko kami yang lain, silakan kunjungi website kami.",
            "Anda dapat menemukan toko terdekat kami di Kota A. Kunjungi website kami untuk detail lokasi toko kami yang lain.",
            "Toko terdekat dari lokasi Anda berada di Kota A. Cek website kami untuk mengetahui lokasi lainnya yang dekat dengan Anda."
        ]
    }
}

# Kamus pertanyaan
questions = {
    "schedule": {
        "en": [
            "When does the store open on weekdays?",
            "Are you open late tonight?",
            "What are your adjusted hours for [Holiday Name]?",
            "I'd like to visit this afternoon. Are you open at [time]?",
            "How early can I come in on [day of the week]?"
        ],
        "id": [
            "Jam berapa toko buka di hari biasa?",
            "Apakah toko buka sampai larut malam ini?",
            "Bagaimana jam operasional khusus untuk [Nama Hari Libur]?",
            "Saya ingin berkunjung sore ini. Apakah toko buka pada pukul [waktu]?",
            "Seberapa awal saya bisa datang pada hari [hari dalam seminggu]?"
        ]
    },
    "promo": {
        "en": [
            "Are there any specific promotions on [product category]?",
            "Do you offer any discounts for new customers?",
            "When does this current promotion end?",
            "How can I apply the promo code I received?",
            "Is there a loyalty program available?"
        ],
        "id": [
            "Apakah ada promo khusus untuk [kategori produk]?",
            "Apakah Anda menawarkan diskon untuk pelanggan baru?",
            "Kapan promosi ini berakhir?",
            "Bagaimana cara menggunakan kode promo yang saya terima?",
            "Apakah ada program loyalitas yang tersedia?"
        ]
    },
    "location": {
        "en": [
            "In which city is your closest store located?",
            "Is there a store near [landmark]?",
            "What is the phone number for your [store location]?",
            "Can I schedule an appointment to visit the store?",
            "Is your store accessible by public transportation?"
        ],
        "id": [
            "Di kota mana toko terdekat Anda berada?",
            "Apakah ada toko di dekat [titik landmark]?",
            "Berapa nomor telepon untuk [lokasi toko]?",
            "Dapatkah saya membuat janji untuk mengunjungi toko?",
            "Apakah toko Anda dapat diakses dengan transportasi umum?"
        ]
    }
}

# Fungsi untuk membersihkan teks
def clean_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english') + stopwords.words('indonesian'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return lemmatized_tokens

# Fungsi untuk mengekstrak fitur dari teks
def extract_features(text):
    features = {}
    for word in clean_text(text):
        features[word] = True
    return features

# Memproses data pelatihan
training_data = []
for intent, questions_dict in questions.items():
    for lang, question_list in questions_dict.items():
        for question in question_list:
            features = extract_features(question)
            training_data.append((features, intent))

# Melatih model Naive Bayes
classifier = NaiveBayesClassifier.train(training_data)

def detect_language(text):
    try:
        lang = detect(text)
    except:
        lang = 'en'  # Default to English if language detection fails
    return lang

def get_answer(question_text):
    lang = detect_language(question_text)
    features = extract_features(question_text)
    intent = classifier.classify(features)
    answers_list = answers[intent][lang]
    random.shuffle(answers_list)
    return answers_list[0]  # Mengembalikan jawaban pertama setelah diacak

# Contoh penggunaan
question = input("Silakan masukkan pertanyaan Anda: ")
lang = detect_language(question)
print(lang)
print(get_answer(question))

# def extract_features(text):
#     features = {}
#     for word in clean_text(text):
#         features[word] = True
#         # Menambahkan fitur untuk kata-kata yang mirip (fuzzy matching) dari semua topik
#         for topic, lang_questions in questions.items():
#             for lang, question_list in lang_questions.items():
#                 similar_words = process.extract(word, clean_text(' '.join(question_list)), limit=3)
#                 for similar_word, _ in similar_words:
#                     features[similar_word] = True
#     return features
