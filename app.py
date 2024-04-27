from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import get_answer

app = Flask(__name__)
CORS(app)  # Tambahkan ini untuk mengaktifkan CORS

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    question = data['question']
    main_topic, answer, recommended_questions = get_answer(question)
    response = {
        'main_topic': main_topic,
        'answer': answer,
        'recommended_questions': recommended_questions
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
