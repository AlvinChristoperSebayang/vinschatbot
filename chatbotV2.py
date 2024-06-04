from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langdetect import detect,LangDetectException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import random
import json
nltk.download('stopwords')
nltk.download('wordnet')
# Load questions and answers from JSON files
with open('questions.json', 'r') as file:
    questions = json.load(file)
with open('answers.json', 'r') as file:
    answers = json.load(file)

custom_stopword = {
    'how', 'can', 'your', 'and', 'to', 'in', 'hi', 'hello', 'hey', 'halo', 'hai', 'apa kabar',
    'how are you', 'how is it going', 'terimakasih', 'thanks', 'thank you', 'dada', 'bye', 'good bye', 'hi', 'hello', 
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
    if not answers_list:
        main_topic = "notFound"
    return main_topic, answers_list[0]


# predicted_intents = []
# actual_intents = []

# # Memperoleh hasil prediksi dan intent sebenarnya untuk setiap pertanyaan dalam test data
# for question_text, actual_intent, _ in test_data:
#     predicted_intent, _ = get_answer(question_text)
#     predicted_intents.append(predicted_intent)
#     actual_intents.append(actual_intent)

# # Membuat confusion matrix
# cm = confusion_matrix(actual_intents, predicted_intents)

# # Menampilkan confusion matrix menggunakan heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=best_pipeline.classes_, yticklabels=best_pipeline.classes_)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()
    


# # Menghitung dan mencetak nilai akurasi, presisi, dan recall
# accuracy = accuracy_score(actual_intents, predicted_intents)
# precision = precision_score(actual_intents, predicted_intents, average='weighted')
# recall = recall_score(actual_intents, predicted_intents, average='weighted')

# print(f'Akurasi: {accuracy:.2f}')
# print(f'Presisi: {precision:.2f}')
# print(f'Recall: {recall:.2f}')
# exit_keywords = ["exit", "quit", "close"]
# while True:
#     question = input("Silakan masukkan pertanyaan Anda atau ketik 'exit' untuk keluar: ")

#     if question.lower() in exit_keywords:
#         print("Terima kasih telah menggunakan layanan kami.")
#         break
#     main_topic, answer = get_answer(question)
#     print("Main Topic/Intent:", main_topic)
#     print("Answer:", answer)

# Contoh teks
# Teks
# text = "what is your services?"
# print('Pertanyaan :', text)

# # Case Folding
# case_folding = text.lower()
# print('Hasil Case Folding:', case_folding)

# # Tokenisasi
# tokens = word_tokenize(case_folding)
# print('Hasil Tokenizing:', tokens)

# # Lematisasi dan Filtering
# lemmatizer = WordNetLemmatizer()
# stop_words = set(stopwords.words('english') + stopwords.words('indonesian')) - custom_stopword
# filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
# lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
# result_lemma = ' '.join(lemmatized_tokens)
# print('Hasil Lemmatization dan Filtering :', result_lemma)

# # Menggunakan TfidfVectorizer untuk menghitung TF-IDF
# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform([result_lemma])

# # Mendapatkan nama fitur (term) dan nilai TF-IDF
# feature_names = vectorizer.get_feature_names_out()
# tfidf_values = tfidf_matrix.toarray()[0]

# # Mencetak hasil TF-IDF
# for term, tfidf in zip(feature_names, tfidf_values):
#     print(f"{term}: {tfidf}")