from flask import Flask, render_template, request, jsonify
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow as tf
import random
import json

app = Flask(__name__)

stemmer = LancasterStemmer()

# Load the model and data
model = tf.keras.models.load_model('chatbot_model.h5')

with open('intents.json') as file:
    intents = json.load(file)

words = [stemmer.stem(w.lower()) for w in
         ["hi", "how", "are", "you", "is", "anyone", "there", "hello", "good", "day", "bye", "see", "later", "goodbye",
          "thanks", "thank", "you", "that's", "helpful"]]
labels = ["greeting", "goodbye", "thanks"]


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['message']
    results = model.predict(np.array([bag_of_words(user_message, words)]))
    results_index = np.argmax(results)
    tag = labels[results_index]

    for tg in intents['intents']:
        if tg['tag'] == tag:
            responses = tg['responses']

    bot_response = random.choice(responses)
    return jsonify({'response': bot_response})


if __name__ == '__main__':
    app.run(debug=True)
