import streamlit as st
import requests

st.title('Chatbot Interface')

def get_response(message):
    url = 'http://127.0.0.1:5000/chatbot'
    payload = {'question': message}
    response = requests.post(url, json=payload)
    return response.json()

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

with st.form(key='chat_form'):
    user_input = st.text_input('You:')
    submit_button = st.form_submit_button(label='Send')

if submit_button and user_input:
    response = get_response(user_input)
    st.session_state['messages'].append(('User', user_input))
    st.session_state['messages'].append(('Bot', response['answer']))
    st.session_state['messages'].append(('Response Time', f"{response['response_time']:.4f} seconds"))

for user, message in st.session_state['messages']:
    st.write(f"{user}: {message}")
