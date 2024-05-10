from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
import random
import json

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

# Function to extract features
def extract_features(text):
    return ' '.join(clean_text(text))

# Prepare training data
training_data = []
for main_topic, sub_topics in questions.items():
    for sub_topic, questions_dict in sub_topics.items():
        for lang, question_list in questions_dict.items():
            for question in question_list:
                features = extract_features(question)
                training_data.append((features, sub_topic))

# Create a pipeline with TfidfVectorizer and MultinomialNB
pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the pipeline
pipeline.fit([data[0] for data in training_data], [data[1] for data in training_data])

# Function to detect language of text
def detect_language(text):
    try:
        lang = detect(text)
    except:
        lang = 'en'
    return lang

# Function to get answer for a question
def get_answer(question_text):
    # Check for greetings
    if any(word in question_text.lower() for word in ["hi", "hello", "hey", "halo", "hai"]):
        return "greetings", "Hello! How can I assist you today?", []

    lang = detect_language(question_text)
    features = extract_features(question_text)
    intent = pipeline.predict([features])[0]
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
        return "notFound", "Sorry, we couldn't find an answer to your question.", []

    return main_topic, answers_list[0], recommended_questions

# Test data
test_data = [
    ("What are your company's core values?", "visionAndMission", "en"),
    ("How can I access your services?", "services", "en"),
    ("Bagaimana budaya kerja di perusahaan Anda?", "cultureCompany", "id"),
    ("Can you introduce me to your team?", "teamCompany", "en"),
    ("Apa rencana masa depan perusahaan Anda?", "futureCompany", "id"),
    ("thank you friend", "gratitude", "en"),
    ("hi, how are you doing?", "greetings", "en"),
    ("apa kabar?", "greetings", "id"),
]

# Prepare test features
test_features = [extract_features(text) for text, _, _ in test_data]

# Prepare actual labels
actual_labels = [label for _, label, _ in test_data]

# Predict labels
predicted_labels = [pipeline.predict([features])[0] for features in test_features]

# Compute confusion matrix
conf_matrix = confusion_matrix(actual_labels, predicted_labels)
print("Confusion Matrix:")
print(conf_matrix)

# Predict labels
predictions = [pipeline.predict([features])[0] for features in test_features]

# Print actual and predicted labels
print("Actual vs Predicted:")
for i, (text, actual_label, _) in enumerate(test_data):
    print(f"{i+1}. Text: {text} | Actual: {actual_label} | Predicted: {predictions[i]}")

# Compute confusion matrix
conf_matrix = confusion_matrix(actual_labels, predicted_labels)
print("\nConfusion Matrix:")
print(conf_matrix)

# Calculate accuracy
accuracy = accuracy_score(actual_labels, predicted_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")



# Calculate accuracy
accuracy = accuracy_score(actual_labels, predicted_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")