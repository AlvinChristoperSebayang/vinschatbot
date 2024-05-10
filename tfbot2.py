import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from langdetect import detect
from nltk.classify import NaiveBayesClassifier
import random
import json
import string
import pandas as pd

with open('questions.json', 'r') as file:
    questions = json.load(file)
with open('answers.json', 'r') as file:
    answers = json.load(file)

def remove_stopwords(text):
    stopwords_list = set(stopwords.words('english') + stopwords.words('indonesian'))
    output = [i for i in text if i not in stopwords_list]
    return output

def stemming(text):
    porter_stemmer = PorterStemmer()
    stem_text = [porter_stemmer.stem(word) for word in text]
    return stem_text

def lemmatizer(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text

def remove_punctuation(text):
    punctuationfree = "".join([i for i in text if i not in string.punctuation])
    return punctuationfree

def clean_text(text):
    text = remove_punctuation(text)
    tokens = nltk.word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum()]
    lemmatized_tokens = lemmatizer(filtered_tokens)
    return lemmatized_tokens

def extract_features(text):
    features = {}
    for word in clean_text(text):
        features[word] = features.get(word, 0) + 1
    return features

training_data = []
for main_topic, sub_topics in questions.items():
    for sub_topic, questions_dict in sub_topics.items():
        for lang, question_list in questions_dict.items():
            for question in question_list:
                features = extract_features(question)
                features['topic'] = main_topic
                training_data.append((features, sub_topic))

classifier = NaiveBayesClassifier.train(training_data)

def detect_language(text):
    try:
        lang = detect(text)
    except:
        lang = 'en'
    return lang

def get_answer(question_text):
    lang = detect_language(question_text)
    features = extract_features(question_text)
    intent = classifier.classify(features)
    main_topic = intent
    answers_list = answers.get(intent, {}).get(lang, [])
    random.shuffle(answers_list)
    
    recommended_questions = []
    for topic, sub_topics in questions.items():
        if topic == main_topic:
            continue
        for sub_topic, questions_dict in sub_topics.items():
            if sub_topic == intent:
                lang_questions = questions_dict.get(lang, [])
                if lang_questions:
                    recommended_questions.append(random.choice(lang_questions))
    
    if not answers_list:
        return "Not Found", "Sorry, we couldn't find an answer to your question.", []
    
    return main_topic, answers_list[0], recommended_questions

exit_keywords = ["exit", "quit", "close"]

while True:
    question = input("Silakan masukkan pertanyaan Anda atau ketik 'exit' untuk keluar: ")

    if question.lower() in exit_keywords:
        print("Terima kasih telah menggunakan layanan kami.")
        break
    main_topic, answer, recommended_questions = get_answer(question)
    print("Main Topic:", main_topic)
    print("Answer:", answer)
    print("Recommended Questions:")
    for i, q in enumerate(recommended_questions, 1):
        print(f"{i}. {q}")