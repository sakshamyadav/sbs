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
    Get Article from URL.

    :param url: string, the url of the article
    :return: (headline, article): (string, string), headline and article texts
    """

    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    p = soup.prettify()

    #find headline text
    end = False
    startindex = p.find('"headline":"')
    i = startindex + 12
    headline = ''
    while True:
        headline += p[i]
        i += 1
        if p[i] == '"':
            break

    #find article text
    startindex = p.find('"articleBody":"')
    end = False
    i = startindex + 15
    article = ''
    while True:
        article += p[i]
        i += 1
        if p[i] == '"':
            break

    #clean up text by removing unwanted html elements
    article = article.replace('amp;', ' ')
    article = article.replace(',', ' ')
    article = article.replace('&', ' ')
    article = article.replace(';', ' ')
    article = article.replace('quot', ' ')
    article = article.replace('lt', ' ')
    article = article.replace('gt', ' ')
    article = article.replace('apos', ' ')
    article = article.replace('x2019', ' ')
    article = article.replace('x2013', ' ')

    return headline, article

#preprocess article by removing stopwords
def preprocess_article(article):
    """
    Preprocess article by removing stopwords

    :param article: string, article text
    :return: list, tokenised list of words in the article
    """

    tokens = article.split()
    tokens = [w.lower() for w in tokens if w.isalpha()]

    tokens = [w for w in tokens if w not in stopwords]

    return tokens

#perform named entity recognition on article
def perform_ner(article):
    """
    Performed named entity recognition on article

    :param article: string, article text
    :return: list, all entities in the article and their respective types
    """

    doc = nlp(article)
    entities = []
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))

    return entities

#summarise article in 'num_sentences' sentences
def summarise_article(article, num_sentences=3):
    """
    Summarise article in 'num_sentences' sentences

    :param article: string, the full text of the article
    :param num_sentences: int, the number of sentences to include in the summary (default=3)
    :return: string, the summary of the article
    """

    #tokenize the article text into sentences
    sentences = sent_tokenize(article)

    words = preprocess_article(article)


    #calculate the word frequency for each word in the article
    word_freq = {}
    for word in words:
        if word not in word_freq:
            word_freq[word] = 1
        else:
            word_freq[word] += 1

    #calculate the sentence score for each sentence
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = word_freq[word]
                else:
                    sentence_scores[sentence] += word_freq[word]

    #get the top N sentences with the highest scores
    summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    #return the summary as a string
    summary = '\n \n'.join(summary_sentences)

    return summary

#analyse sentiment in headline by extracting polarity and subjectivity
def analyse_sentiment(headline):
    """
    Analyse sentiment in headline by extracting polarity and subjectivity.

    :param headline: string, article headline text
    :return: (polarity, subjectivity): (int,int), polarity and subjectivity of the article headline
    """

    headline = headline.split()
    headline = [h.lower() for h in headline if h.isalpha()]
    headline = ' '.join(headline)
    
    blob = TextBlob(headline)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    return polarity, subjectivity

#create main streamlit form to accept inputs
with st.form("main form"):
    url = st.text_input('Enter SBS News Article URL in format: https://www.sbs.com.au/news/article/...', value="https://www.sbs.com.au/news/article/are-you-being-spied-on-by-a-foreign-government-australian-federal-police-want-to-hear-from-you/t5e2srqo1")
    md_categories = ['Named Entity Recognition', 'Sentiment Analysis', 'Article Summariser (top 3 sentences)']
    nlp_task = st.selectbox("Select the type of metadata you would like to extract", md_categories)

    submitted = st.form_submit_button("Submit")

#choose NLP algorithm based on user selection
if submitted:
    headline, article = get_article(url)
    st.header(headline)
    if nlp_task == "Named Entity Recognition":
        entities = perform_ner(article)
        st.write(entities)

    elif nlp_task == "Sentiment Analysis":
        p, s = analyse_sentiment(headline)
        st.write(f"Polarity (-1 to 1 indicating negativity to positivity): {p:.2f}")
        st.write(f"Subjectivity (0 to 1 indicating objectivity to subjectivity): {s:.2f}")

    elif nlp_task == "Article Summariser (top 3 sentences)":
        summary = summarise_article(article)
        st.write(summary)
