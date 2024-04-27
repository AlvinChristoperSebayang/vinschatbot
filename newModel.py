from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
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
    return " ".join(lemmatized_tokens)

def extract_features(text):
    features = {}
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([text])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    for idx, feature_name in enumerate(feature_names):
        features[feature_name] = tfidf_matrix[0, idx]
    return features

# Proses data latih
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
    
    if not answers_list:
        return main_topic, "Jawaban tidak ditemukan", []
    
    random.shuffle(answers_list)
    
    recommended_questions = []
    for topic, sub_topics in questions.items():
        if topic == main_topic:
            continue
        for sub_topic, questions_dict in sub_topics.items():
            if sub_topic == intent:
                continue
            recommended_questions.append(random.choice(questions_dict.get(lang, [])))
    
    return main_topic, answers_list[0], recommended_questions
