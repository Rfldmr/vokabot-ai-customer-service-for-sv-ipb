import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from tensorflow import keras
import json
import random
import streamlit as st
import time

model = keras.models.load_model('Model/vokabot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('Model/words.pkl','rb'))
classes = pickle.load(open('Model/classes.pkl','rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = np.zeros(len(words), dtype=np.float32)
    for sw in sentence_words:
        for i, word in enumerate(words):
            if word == sw:
                bag[i] = 1
    return bag

def predict_class(sentence):
    p = bag_of_words(sentence, words)
    res = model.predict(np.expand_dims(p, axis=0))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def getResponse(ints, intents_json):
    if not ints:
        return "Maaf, aku tidak mengerti."
    tag = classes[ints[0][0]]
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            return result
    return "Maaf, aku tidak menemukan jawaban untuk itu."



st.title("Selamat Datang di VokaBot!")

st.sidebar.title("👾 VokaBot AI - (Beta)")
option = st.sidebar.selectbox(
    "",
    ("VokaBot AI - (Beta v.0.0.2)"),
)
st.sidebar.markdown("")
st.sidebar.markdown("VokaBot merupakan customer service berbasis AI yang selalu siap 24/7 untuk membantu memberikan informasi dan menjawab masalah para mahasiswa.")
st.sidebar.markdown("")
st.sidebar.markdown("[Pelajari Selengkapnya](https://docs.streamlit.io/develop/api-reference/layout/st.columns)")

st.sidebar.markdown("---")
st.sidebar.warning("VokaBot AI **tidak dirancang** untuk menyimpan data yang diinput oleh pengguna, menjamin sistem bebas dari kemungkinan pencurian data.")
st.sidebar.markdown("")
if st.sidebar.button("Reset Percakapan"):
    st.session_state.messages = [{"role": "assistant", "content": "Hai! Ada yang bisa kubantu hari ini?"}]


st.markdown("")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hai! Ada yang bisa kubantu hari ini?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Tanyakan sesuatu, misal 'Bagaimana cara mengajukan cuti?'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Sedang memikirkan jawaban..."):
            time.sleep(1)
            ints = predict_class(prompt)
            response = getResponse(ints, intents)
            message_placeholder.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})