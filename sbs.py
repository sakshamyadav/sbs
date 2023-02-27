import requests
from bs4 import BeautifulSoup
import spacy
import streamlit as st
import nltk

nltk.download('stopwords')
nltk.download

with st.form("articleurl"):
    url = st.text_input('Enter SBS News Article URL')
    submitted = st.form_submit_button("Submit")

if submitted:    
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    p = soup.prettify()
    startindex = p.find('"articleBody":"')

    end = False
    i = startindex+15

    article = ''
    while True:
        article += p[i]
        i+=1
        if p[i] == '"':
            break

    article = article.replace('amp;',' ')
    article=article.replace('.', ' ')
    article=article.replace(',', ' ')
    article=article.replace('&', ' ')
    article=article.replace(';', ' ')

    stopwords = nltk.corpus.stopwords.words('english')
    article = article.split()
    article = [w.lower() for w in article if w.isalpha()]
    stopwords = nltk.corpus.stopwords.words('english')

    article = [w for w in article if w not in stopwords]

    st.write(article)


