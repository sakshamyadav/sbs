#import required libraries
import requests
from bs4 import BeautifulSoup
import spacy
import streamlit as st
import nltk
from textblob import TextBlob
from nltk.tokenize import word_tokenize, sent_tokenize
from heapq import nlargest

#download nltk resources
nltk.download('stopwords')
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")
stopwords = nltk.corpus.stopwords.words('english')

#title of the web app
st.title("Metadata Generation from SBS Articles")

#get article from url
def get_article(url):
    """
    Summarize an article by extracting the most important sentences.

    :param url: string, the url of the article
    :return: (headline, article): (string, string), headline and article texts
    """

    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    p = soup.prettify()

    end = False
    startindex = p.find('"headline":"')

    i = startindex + 12
    headline = ''
    while True:
        headline += p[i]
        i += 1
        if p[i] == '"':
            break

    startindex = p.find('"articleBody":"')

    end = False
    i = startindex + 15

    article = ''
    while True:
        article += p[i]
        i += 1
        if p[i] == '"':
            break

    article = article.replace('amp;', ' ')
    article = article.replace('.', ' ')
    article = article.replace(',', ' ')
    article = article.replace('&', ' ')
    article = article.replace(';', ' ')
    article = article.replace('quot', ' ')


    return headline, article


def preprocess_article(article):
    """
    Summarize an article by extracting the most important sentences.

    :param article: string, article text
    :return: list, tokenised list of words in the article
    """

    tokens = article.split()
    tokens = [w.lower() for w in tokens if w.isalpha()]

    tokens = [w for w in tokens if w not in stopwords]

    return tokens


def perform_ner(article):
    """
    Summarize an article by extracting the most important sentences.

    :param article: string, article text
    :return: list, all entities in the article and their respective types
    """

    doc = nlp(article)
    entities = []
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))

    return entities


def summarize_article(article, num_sentences=3):
    """
    Summarize an article by extracting the most important sentences.

    :param article: string, the full text of the article
    :param num_sentences: int, the number of sentences to include in the summary (default=3)
    :return: string, the summary of the article
    """

    # Tokenize the article text into sentences
    sentences = sent_tokenize(article)


    # Calculate the word frequency for each word in the article
    word_freq = {}
    for word in word_tokenize(article.lower()):
        if word not in stopwords and word.isalpha():
            if word not in word_freq:
                word_freq[word] = 1
            else:
                word_freq[word] += 1

    # Calculate the sentence score for each sentence
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = word_freq[word]
                else:
                    sentence_scores[sentence] += word_freq[word]

    # Get the top N sentences with the highest scores
    summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)

    # Return the summary as a string
    summary = ' '.join(summary_sentences)

    return summary


def analyse_sentiment(headline):
    """
    Summarize an article by extracting the most important sentences.

    :param headline: string, article headline text
    :return: (polarity, subjectivity): (int,int), polarity and subjectivity of the article headline
    """

    blob = TextBlob(headline)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    return polarity, subjectivity

with st.form("main form"):
    url = st.text_input('Enter SBS News Article URL')
    md_categories = ['Named Entity Recognition', 'Sentiment Analysis', 'Article Summariser']
    nlp_task = st.selectbox("Select the type of metadata you would like to extract", md_categories)

    submitted = st.form_submit_button("Submit")

if submitted:
    headline, article = get_article(url)
    st.header(headline)
    if nlp_task == "Named Entity Recognition":
        ents = perform_ner(article)
        st.write(ents)

    elif nlp_task == "Sentiment Analysis":
        p, s = analyse_sentiment(headline)
        st.write(f"Polarity: {p:.2f}")
        st.write(f"subjectivity: {s:.2f}")

    elif nlp_task == "Article Summariser":
        out = summarize_article(article)
        st.write(out)