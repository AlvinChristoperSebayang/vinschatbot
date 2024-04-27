from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langdetect import detect
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import distance
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

def extract_features(text):
    features = {}
    for word in clean_text(text):
        features[word] = features.get(word, 0) + 1
    return features

def get_best_matching_question(question, questions_list):
    best_match = None
    best_similarity = 0
    for q in questions_list:
        sim = distance.edit_distance(question, q)
        if best_match is None or sim < best_similarity:
            best_match = q
            best_similarity = sim
    return best_match, best_similarity

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
        best_question, similarity = get_best_matching_question(question_text, [q for q in questions.get(main_topic, {}).get(intent, {}).get(lang, [])])
        if similarity < 0.5:
            return "Not Found", "Sorry, we couldn't find an answer to your question.", [], similarity
        else:
            return main_topic, f"Sorry, we couldn't find an exact answer to your question. Did you mean '{best_question}'?", [], similarity
    
    return main_topic, answers_list[0], recommended_questions, None


exit_keywords = ["exit", "quit", "close"]

while True:
    question = input("Silakan masukkan pertanyaan Anda atau ketik 'exit' untuk keluar: ")

    if question.lower() in exit_keywords:
        print("Terima kasih telah menggunakan layanan kami.")
        break
    main_topic, answer, recommended_questions, similarity = get_answer(question)
    print("Main Topic:", main_topic)
    print("Answer:", answer)
    print("Recommended Questions:")
    for i, q in enumerate(recommended_questions, 1):
        print(f"{i}. {q}")
    if similarity is not None:
        print("Similarity:", similarity)
