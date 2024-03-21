import sys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.classify import NaiveBayesClassifier
from langdetect import detect

# Contoh data latih untuk kedua bahasa
training_data = [
    ("When does the store open?", "schedule"),
    ("Are there any discounts today?", "promo"),
    ("Where is the store located?", "location"),
    ("What are the latest products sold?", "products"),
    ("How do I return an item?", "service"),
    ("Kapan toko buka?", "schedule"),
    ("Apakah ada diskon hari ini?", "promo"),
    ("Di mana lokasi toko?", "location"),
    ("Apa produk terbaru yang dijual?", "products"),
    ("Bagaimana cara mengembalikan barang?", "service")
]
# Contoh data latih untuk kedua bahasa


# Praproses teks untuk kedua bahasa
stop_words_en = set(stopwords.words('english'))
stop_words_id = set(stopwords.words('indonesian'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text, language):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
    
    if language == 'en':
        stop_words = stop_words_en
        language_text = "bahasa Inggris"
    elif language == 'id':
        stop_words = stop_words_id
        language_text = "bahasa Indonesia"
    else:
        stop_words = set()
        language_text = "bahasa tidak terdeteksi kami hanya memahami bahasa indonesia dan bahasa inggris/The language is not detected. We only understand Indonesian and English."

    filtered_tokens = [token for token in filtered_tokens if token not in stop_words]
    return dict([(token, True) for token in filtered_tokens]), language_text

# Ekstraksi fitur dan pelatihan model Naive Bayes
training_features = [(preprocess_text(text, detect(text))[0], label) for (text, label) in training_data]
classifier = NaiveBayesClassifier.train(training_features)

# Input pertanyaan dari terminal
question = input("Masukkan pertanyaan: ")
# Dictionary jawaban untuk setiap label
answers = {
    "schedule": "Toko buka pada jam 08.00 - 22.00 setiap hari.",
    "promo": "Hari ini ada diskon 20% untuk semua produk.",
    "location": "Toko kami berlokasi di Jl. Jendral Sudirman No. 123, Jakarta.",
    "products": "Produk terbaru kami adalah sepatu olahraga dan tas ransel.",
    "service": "Untuk mengembalikan barang, silakan kunjungi halaman 'Pengembalian' di situs kami."
}

# Klasifikasi pertanyaan
preprocessed_question, language = preprocess_text(question, detect(question))
predicted_label = classifier.classify(preprocessed_question)

# Tampilkan jawaban sesuai dengan prediksi label
print("Bahasa pertanyaan:", language)
print("Topik pertanyaan:", predicted_label)
print("Jawaban:", answers.get(predicted_label, "Maaf, saya tidak mengerti pertanyaan Anda."))
