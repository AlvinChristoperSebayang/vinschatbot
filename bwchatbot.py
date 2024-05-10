from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langdetect import detect
from nltk.classify import NaiveBayesClassifier
import random
import json
from collections import Counter

# Load questions and answers from JSON files
with open('questions.json', 'r') as file:
    questions = json.load(file)
with open('answers.json', 'r') as file:
    answers = json.load(file)

# Function to clean text
def clean_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english') + stopwords.words('indonesian'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return lemmatized_tokens

# Function to extract bag of words features
def extract_features(text, vocabulary):
    tokens = clean_text(text)
    features = Counter(tokens)
    # Bag of Words representation
    bow = [features[word] if word in features else 0 for word in vocabulary]
    return bow

# Build vocabulary
all_words = []
for main_topic, sub_topics in questions.items():
    for sub_topic, questions_dict in sub_topics.items():
        for lang, question_list in questions_dict.items():
            for question in question_list:
                all_words.extend(clean_text(question))

vocabulary = set(all_words)

# Build training data
training_data = []
for main_topic, sub_topics in questions.items():
    for sub_topic, questions_dict in sub_topics.items():
        for lang, question_list in questions_dict.items():
            for question in question_list:
                features = extract_features(question, vocabulary)
                features.append(main_topic)  # Add topic as a feature
                training_data.append((dict(zip(vocabulary, features[:-1])), sub_topic))

# Classifier training
classifier = NaiveBayesClassifier.train(training_data)

# Function to detect language of text
def detect_language(text):
    try:
        lang = detect(text)
    except:
        lang = 'en'
    return lang

# Function to get answer to a question
def get_answer(question_text):
    lang = detect_language(question_text)
    features = extract_features(question_text, vocabulary)
    intent = classifier.classify(dict(zip(vocabulary, features)))
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

# Main loop
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
