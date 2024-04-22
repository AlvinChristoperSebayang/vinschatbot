import nltk
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.classify import NaiveBayesClassifier

# Read JSON data
with open('questions.json', 'r') as file:
    data = json.load(file)

# Preprocessing
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return dict([(word, True) for word in filtered_tokens])

# Format data for training
training_data = []
for intent, intent_data in data.items():
    for lang, questions in intent_data.items():
        for question in questions:
            training_data.append((preprocess_text(question), intent))

# Train the classifier
classifier = NaiveBayesClassifier.train(training_data)

# Chatbot loop
while True:
    user_input = input("User: ")
    if user_input.lower() == 'exit':
        break

    
    processed_input = preprocess_text(user_input)


    predicted_intent = classifier.classify(processed_input)
    responses = data.get(predicted_intent, {}).get('en', ['I am sorry, but I do not understand.'])
    response = responses[0]

    print(f"Bot: {response}")
