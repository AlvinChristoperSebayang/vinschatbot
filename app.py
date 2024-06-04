from flask import Flask, request, jsonify
from flask_cors import CORS
import time
from chatbotV2 import get_answer

app = Flask(__name__)
CORS(app)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    start_time = time.time() 
    data = request.get_json()
    question = data['question']
    main_topic, answer = get_answer(question)
    end_time = time.time()  
    response_time = end_time - start_time  
    
    response = {
        'main_topic': main_topic,
        'answer': answer,
        'response_time': response_time  
    }
    
    # Log waktu respons
    print(f"Response Time: {response_time:.4f} seconds")
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
