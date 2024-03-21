import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize

# Data training
training_data = [
    ("What is your name?", "My name is Chatbot."),
    ("How are you?", "I'm good, thank you."),
    ("What can you do?", "I can answer your questions."),
]

# Tokenisasi dan preprocessing
tokenizer = Tokenizer()
tokenizer.fit_on_texts([pair[0] for pair in training_data])

# Membuat model LSTM
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, 64),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(len(tokenizer.word_index) + 1, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Melatih model
X = tokenizer.texts_to_sequences([pair[0] for pair in training_data])
X = pad_sequences(X, padding='post')
y = tokenizer.texts_to_sequences([pair[1] for pair in training_data])
y = pad_sequences(y, padding='post')

model.fit(X, y, epochs=100, verbose=1)

# Fungsi chatbot
def chat():
    print("Hello! Ask me anything or say 'bye' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'bye':
            break
        input_sequence = tokenizer.texts_to_sequences([user_input])
        padded_input_sequence = pad_sequences(input_sequence, padding='post')
        predicted_sequence = model.predict(padded_input_sequence)
        predicted_word_index = tf.argmax(predicted_sequence, axis=-1).numpy()[0]
        response = tokenizer.index_word[predicted_word_index]
        print("Bot:", response)

# Mulai chatbot
chat()
