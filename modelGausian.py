from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
import random
import json

# Load questions and answers from JSON files
with open('questions.json', 'r') as file:
    questions = json.load(file)
with open('answers.json', 'r') as file:
    answers = json.load(file)

custom_stopword = {
    'how', 'can', 'your', 'are', 'the', 'is', 'of', 'and', 'to', 'in', 'hi', 'hello', 'hey', 'halo', 'hai', 'apa kabar',
    'how are you', 'how is it going', 'terimakasih', 'thanks', 'thank you', 'dada', 'bye', 'goodbye', 'hi', 'hello', 
    'halo', 'hai', 'selamat pagi', 'selamat siang', 'selamat sore', 'selamat malam', 'tentang', 'about',   "bye","goodbye", "see you later","take care","dada"
}

# Function to clean text without removing stopwords
def clean_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english') + stopwords.words('indonesian')) - custom_stopword
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

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'multinomialnb__alpha': [0.1, 0.5, 1.0]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit([data[0] for data in training_data], [data[1] for data in training_data])
best_pipeline = grid_search.best_estimator_

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
    question_words = set(clean_text(question_text))
    training_words = set(word for data in training_data for word in clean_text(data[0]))
    common_words = question_words.intersection(training_words)

    if len(common_words) < 1:
        intent = "notFound"
    else:
        intent = best_pipeline.predict([features])[0]

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
    ("Can you introduce me to your team?", "teamCompany"),
    ("Apa rencana masa depan perusahaan Anda?", "futureCompany"),
    ("thank you friend", "gratitude"),
    ("jelaskan mengenai mangcoding", "aboutMangcoding"),
    ("jelaskan mengenai sayur mayur", "notFound"),
    ("what is your services", "services"),
    ("jelaskan sejarah perusahaan anda?" , "historyCompany"),
    ("tell me about your company history" , "historyCompany"),
    ( "What is the history of your company?", "historyCompany"),
    ( "apa yang menjadi misi perusahaan anda", "visionAndMission"),
    ( "Apa visi dan misi perusahaan Anda?", "visionAndMission"),
    ( "bye", "visionAndMission"),
    ( "", "visionAndMission"),
    ( "jelaskan visi dan misi perusahaan anda?", "visionAndMission"),

]

# Get predicted labels using get_answer function
predicted_labels = [get_answer(text)[0] for text, _ in training_data]

# Prepare actual labels
actual_labels = [label for _, label in training_data]

# Print actual and predicted labels
print("Actual vs Predicted:")
for i, (text, actual_label) in enumerate(training_data):
    print(f"{i+1}. Text: {text} | Actual: {actual_label} | Predicted: {predicted_labels[i]}")

# Compute confusion matrix
conf_matrix = confusion_matrix(actual_labels, predicted_labels)
print("\nConfusion Matrix:")
print(conf_matrix)

# Calculate accuracy
accuracy = accuracy_score(actual_labels, predicted_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")


# # Get predicted labels for training data using get_answer function
# predicted_labels_train = [get_answer(text)[0] for text, _ in training_data]

# # Prepare actual labels for training data
# actual_labels_train = [label for _, label in training_data]

# # Compute confusion matrix for training data
# conf_matrix_train = confusion_matrix(actual_labels_train, predicted_labels_train)
# print("\nConfusion Matrix (Training Data):")
# print(conf_matrix_train)

# # Calculate accuracy for training data
# accuracy_train = accuracy_score(actual_labels_train, predicted_labels_train)
# print(f"Accuracy (Training Data): {accuracy_train * 100:.2f}%")

# # Print actual and predicted labels for training data
# print("\nActual vs Predicted (Training Data):")
# for i, (_, actual_label) in enumerate(training_data):
#     print(f"{i+1}. Actual: {actual_label} | Predicted: {predicted_labels_train[i]}")