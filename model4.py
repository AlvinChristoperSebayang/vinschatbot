import nltk
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.classify import NaiveBayesClassifier
import random
from langdetect import detect
from fuzzywuzzy import process
# Kamus jawaban
with open('questions.json', 'r', encoding='utf-8') as f:
    questions = json.load(f)

with open('answers.json', 'r', encoding='utf-8') as f:
    answers = json.load(f)


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
        # Menambahkan fitur untuk kata-kata yang mirip (fuzzy matching) dari semua topik
        for topic, lang_questions in questions.items():
            for lang, question_list in lang_questions.items():
                similar_words = process.extract(word, clean_text(' '.join(question_list)), limit=3)
                for similar_word, _ in similar_words:
                    features[similar_word] = True
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
        if lang not in ['en', 'id']:
            lang = 'en'  # Default to English for other languages
    except:
        lang = 'en'  # Default to English if language detection fails
    return lang


def get_answer(question_text):
    lang = detect_language(question_text)
    # Jika panjang kata kurang dari 2, kembalikan pesan untuk meminta pertanyaan yang lebih jelas
    if len(question_text.split()) < 2:
        return "Mohon maaf, pertanyaan Anda terlalu singkat. Mohon berikan pertanyaan yang lebih jelas."
    
    features = extract_features(question_text)
    intent = classifier.classify(features)
    answers_list = answers[intent][lang]
    random.shuffle(answers_list)
    return answers_list[0]  # Mengembalikan jawaban pertama setelah diacak

# Contoh penggunaan
question = input("Silakan masukkan pertanyaan Anda: ")
lang = detect_language(question)
print(lang)
print(intent)
print(get_answer(question))
