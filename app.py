import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform_m(m):
    m = m.lower()
    m = nltk.word_tokenize(m)

    y= []
    for i in m:
        if i.isalnum():
            y.append(i)
    m = y[:]
    y.clear()

    for i in m:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    m = y[:]
    y.clear()

    for i in m:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('SMS SPAM CLASSIFIER')
input = st.text_area('Enter your SMS')

if st.button('Predict'):
    transformed_sms = transform_m(input)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    if result == 1 :
        st.header("SPAM")
    else:
        st.header("NOT SPAM")