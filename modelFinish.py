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
    greetings_words = ["hi", "hello", "hey", "halo", "hai", "apa kabar", "how are you", "how is it going"]
    gratitude_words = ["terimakasih","thanks","thank you"]
    if any(word in question_text.lower() for word in greetings_words):
        if "apa kabar" in question_text.lower():
            return "greetings", "Apa kabar? Bagaimana saya bisa membantu Anda hari ini?", []
        elif "how are you" in question_text.lower():
            return "greetings", "How are you? How can I assist you today?", []
        else:
            return "greetings", "Hello! How can I assist you today?", []

    lang = detect_language(question_text)
    features = extract_features(question_text) 
    question_words = set(clean_text(question_text))
    training_words = set(word for data in training_data for word in clean_text(data[0]))
    common_words = question_words.intersection(training_words)


    if not common_words:
        print('empty')    

    if len(common_words) < 1:
            if any(word in question_text.lower() for word in gratitude_words):
                intent = "gratitude"
            else:
                intent = "notFound"
    else:
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

    if not answers_list or len(common_words) < 1:
        return "notFound", "Sorry, I couldn't find a relevant answer for your question. Please try rephrasing or asking something else.", []

    return main_topic, answers_list[0], recommended_questions



# Test data
test_data = [
    ("What are your company's core values?", "visionAndMission"),
    ("How can I access your services?", "services"),
    ("Bagaimana budaya kerja di perusahaan Anda?", "cultureCompany"),
    ("Can you introduce me to your team?", "teamCompany"),
    ("Apa rencana masa depan perusahaan Anda?", "futureCompany"),
    ("thank you friend", "gratitude"),
    ("jelaskan mengenai mangcoding","aboutMangcoding"),
    ("jelaskan mengenai sayur mayur","notFound"),
    ("apa layanan anda?", "services")
]

# Get predicted labels using get_answer function
predicted_labels = [get_answer(text)[0] for text, _ in test_data]

# Prepare actual labels
actual_labels = [label for _, label in test_data]

# Print actual and predicted labels
print("Actual vs Predicted:")
for i, (text, actual_label) in enumerate(test_data):
    print(f"{i+1}. Text: {text} | Actual: {actual_label} | Predicted: {predicted_labels[i]}")

# Compute confusion matrix
conf_matrix = confusion_matrix(actual_labels, predicted_labels)
print("\nConfusion Matrix:")
print(conf_matrix)

# Calculate accuracy
accuracy = accuracy_score(actual_labels, predicted_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")

# CM Training Data

# # Prediksi label dari data training menggunakan fungsi get_answer
# predicted_train_labels = [get_answer(data[0])[0] for data in training_data]

# # Siapkan label aktual
# actual_train_labels = [data[1] for data in training_data]

# # Hitung confusion matrix
# conf_matrix_train = confusion_matrix(actual_train_labels, predicted_train_labels)
# print("\nConfusion Matrix (Training Data):")
# print(conf_matrix_train)

# # Hitung akurasi
# accuracy_train = accuracy_score(actual_train_labels, predicted_train_labels)
# print(f"Accuracy (Training Data): {accuracy_train * 100:.2f}%")

# # (Opsional) Menampilkan label aktual dan prediksi untuk data training
# print("\nActual vs Predicted (Training Data):")
# for i, (text, actual_label) in enumerate(zip(training_data, actual_train_labels)):
#     print(f"{i+1}. Text: {text[0]} | Actual: {actual_label} | Predicted: {predicted_train_labels[i]}")