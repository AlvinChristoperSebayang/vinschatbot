from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langdetect import detect
from nltk.classify import NaiveBayesClassifier
import random
import json

with open('questions.json', 'r') as file:
    questions = json.load(file)
with open('answers.json', 'r') as file:
    answers = json.load(file)

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
for main_topic, sub_topics in questions.items():
    for sub_topic, questions_dict in sub_topics.items():
        for lang, question_list in questions_dict.items():
            for question in question_list:
                features = extract_features(question)
                features['topic'] = main_topic  # Menambahkan main topic sebagai fitur
                training_data.append((features, sub_topic))

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
    main_topic = intent
    answers_list = answers.get(intent, {}).get(lang, [])
    
    # Jika intent tidak ditemukan, kembalikan pesan bahwa jawaban tidak ditemukan
    if not answers_list:
        return main_topic, "Jawaban tidak ditemukan", []
    
    random.shuffle(answers_list)
    
    # Memilih satu pertanyaan dari setiap sub topik selain sub topik dari intent yang sedang ditanyakan
    recommended_questions = []
    for topic, sub_topics in questions.items():
        if topic == main_topic:
            continue
        for sub_topic, questions_dict in sub_topics.items():
            if sub_topic == intent:
                continue
            recommended_questions.append(random.choice(questions_dict.get(lang, [])))
    
    return main_topic, answers_list[0], recommended_questions

# while True:
#     question = input("Silakan masukkan pertanyaan Anda atau ketik 'exit' untuk keluar: ")

#     # Keluar dari loop jika pengguna memasukkan kata kunci keluar
#     if question.lower() in exit_keywords:
#         print("Terima kasih telah menggunakan layanan kami.")
#         break
#     main_topic, answer, recommended_questions = get_answer(question)
#     print("Main Topic:", main_topic)
#     print("Answer:", answer)
#     print("Recommended Questions:")
#     for i, q in enumerate(recommended_questions, 1):
#         print(f"{i}. {q}")

