import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
 
# load the saved vectorizer and naive model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

#transfrom_text function for text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# nltk.download('stopwords')
ps = PorterStemmer()

def transform_text(text):
    text = text.lower() #coverting to lowercase
    text = nltk.word_tokenize(text) #tokenize

    #removing special characters and retaining alphanumeric words
    text = [word for word in text if word.isalnum()]

    #removing stopwords and punctuations
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

    #applying stemming
    text = [ps.stem(word) for word in text]
    return " ".join(text) 


#Streamlit code 
st.title('Email Spam Classifier')
input_sms = st.text_area('Enter the message')

if st.button('Predict'):
    #preprocess
    transformed_sms = transform_text(input_sms)

    #vectorize
    vector_input = tfidf.transform([transformed_sms])

    #predict
    result = model.predict(vector_input)[0]

    # display the result
    if result == 0:
        st.header('Not Spam')
    else:
        st.header('Spam')