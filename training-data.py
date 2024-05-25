from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import random

# Data Preprocessing
# Consider adding stopwords removal and additional text cleaning steps here

# Feature Engineering
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Using TF-IDF with bi-grams

# Model Training
X = [text for text, _ in training_data]
y = [label for _, label in training_data]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Model Optimization
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Model Evaluation
y_pred = model.predict(X_test_vectorized)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Prediction Example
def get_answer(text):
    vectorized_text = vectorizer.transform([text])
    predicted_label = model.predict(vectorized_text)[0]
    return predicted_label

# Test the model with sample data
test_data = [
    ("What are your company's core values?", "visionAndMission"),
    ("How can I access your services?", "services"),
    ("Bagaimana budaya kerja di perusahaan Anda?", "cultureCompany"),
    # Add more test data here
]

for text, actual_label in test_data:
    predicted_label = get_answer(text)
    print(f"Text: {text} | Actual: {actual_label} | Predicted: {predicted_label}")