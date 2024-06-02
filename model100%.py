from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langdetect import detect, LangDetectException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import random
import json
import pandas as pd
import numpy as np

# Load questions and answers from JSON files
with open('questions.json', 'r') as file:
    questions = json.load(file)
with open('answers.json', 'r') as file:
    answers = json.load(file)

custom_stopword = {
    'how', 'can', 'your', 'are', 'the', 'is', 'of', 'and', 'to', 'in', 'hi', 'hello', 'hey', 'halo', 'hai', 'apa kabar',
    'how are you', 'how is it going', 'terimakasih', 'thanks', 'thank you', 'dada', 'bye', 'goodbye', 'hi', 'hello', 
    'halo', 'hai', 'selamat pagi', 'selamat siang', 'selamat sore', 'selamat malam', 'tentang', 'about', 'bye', 'goodbye', 
    'see you later', 'take care', 'dada'
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
        if lang not in ['id', 'en']:
            lang = 'en'
    except LangDetectException:
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

# Prepare training data
X_train = [data[0] for data in training_data]
y_train = [data[1] for data in training_data]

# List to store misclassified questions
misclassified = []

# Predict using the get_answer function for training data
y_pred_train = []
for question in X_train:
    pred_label = get_answer(question)[0]
    y_pred_train.append(pred_label)
    true_label = y_train[X_train.index(question)]
    if pred_label != true_label:
        misclassified.append((question, true_label, pred_label))

# Create confusion matrix for training data
cm_train = confusion_matrix(y_train, y_pred_train)

# Calculate accuracy for training data
accuracy_train = accuracy_score(y_train, y_pred_train)

# Calculate precision for training data
precision_train = precision_score(y_train, y_pred_train, average='weighted')

# Display confusion matrix for training data
plt.figure(figsize=(16, 14))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', xticklabels=best_pipeline.classes_, yticklabels=best_pipeline.classes_, annot_kws={"size": 14})
plt.xlabel('Predicted Label', fontsize=16)
plt.ylabel('True Label', fontsize=16)
plt.title('Confusion Matrix (Training Data)', fontsize=18)
plt.xticks(rotation=30, fontsize=14)
plt.yticks(rotation=45, fontsize=14)
plt.show()

# Display accuracy, precision
print(f'Accuracy (Training Data): {accuracy_train}')
print(f'Precision (Training Data): {precision_train}')
# Function to calculate TP, TN, FP, FN from confusion matrix
def calculate_metrics(cm):
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (TP + FP + FN)
    return TP.sum(), TN.sum(), FP.sum(), FN.sum()

total_TP_train, total_TN_train, total_FP_train, total_FN_train = calculate_metrics(cm_train)

print(f'Total TP (Training Data): {total_TP_train}, Total TN (Training Data): {total_TN_train}, Total FP (Training Data): {total_FP_train}, Total FN (Training Data): {total_FN_train}')

# Display misclassified questions
print("\nMisclassified Questions:")
for question, true_label, pred_label in misclassified:
    print(f"Question: {question}")
    print(f"True Label: {true_label}, Predicted Label: {pred_label}\n")
