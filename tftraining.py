from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langdetect import detect
from nltk.classify import NaiveBayesClassifier
import random
import json
from sklearn.metrics import confusion_matrix
from nltk.util import ngrams

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

# Function to extract features from text
def extract_features(text, n=2):
    features = {}
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english') + stopwords.words('indonesian'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Generate unigrams and bigrams
    unigrams = [token for token in lemmatized_tokens if len(token) == 1]
    bigrams = [' '.join(gram) for gram in ngrams(lemmatized_tokens, n) if len(gram) == n]

    # Combine unigrams and bigrams
    all_tokens = unigrams + bigrams

    for token in all_tokens:
        features[token] = features.get(token, 0) + 1
    return features

# Prepare training data for classifier
training_data = []
for main_topic, sub_topics in questions.items():
    for sub_topic, questions_dict in sub_topics.items():
        for lang, question_list in questions_dict.items():
            for question in question_list:
                features = extract_features(question)
                features['topic'] = main_topic
                training_data.append((features, sub_topic))

# Train the Naive Bayes classifier
classifier = NaiveBayesClassifier.train(training_data)

# Function to detect language of text
def detect_language(text):
    try:
        lang = detect(text)
    except:
        lang = 'en'
    return lang

# Function to get answer for a question
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
# Data uji yang diperbaiki dengan sub-topik dan bahasa
test_data = [
    ("What are your company's core values?", "visionAndMission", "en"),
    ("How can I access your services?", "services", "en"),
    ("Bagaimana budaya kerja di perusahaan Anda?", "cultureCompany", "id"),
    ("Can you introduce me to your team?", "teamCompany", "en"),
    ("Apa rencana masa depan perusahaan Anda?", "futureCompany", "id"),
    ("thank you friend","gratitude","en"),
    ("hi, how are you doing?","greetings","en"),

]


# List untuk menyimpan prediksi chatbot
predictions = []

# List untuk menyimpan kategori yang seharusnya
actual = []

# Melakukan prediksi untuk setiap pertanyaan pada data uji
for question, category, lang in test_data:
    main_topic, _, _ = get_answer(question)
    predictions.append(main_topic)
    actual.append(category)

# Membuat matriks kebingungan
conf_matrix = confusion_matrix(actual, predictions)

# Menampilkan matriks kebingungan
print("Confusion Matrix:")
print(conf_matrix)

# Menghitung akurasi
accuracy = (conf_matrix.diagonal().sum() / conf_matrix.sum()) * 100
print(f"Accuracy: {accuracy:.2f}%")

print(predictions)
print(actual)