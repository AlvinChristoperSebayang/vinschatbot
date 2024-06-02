from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbotV2 import get_answer

app = Flask(__name__)
CORS(app)  

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    question = data['question']
    main_topic, answer = get_answer(question)
    response = {
        'main_topic': main_topic,
        'answer': answer
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
