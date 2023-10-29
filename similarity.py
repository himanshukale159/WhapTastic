import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')


# Function for text preprocessing
def text_preprocessing(text):
    # Convert text to lowercase
    text = text.lower()

    # Tokenization
    words = word_tokenize(text)

    # Remove special characters and numbers
    words = [word for word in words if word.isalpha()]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Stemming (you can also use lemmatization if you prefer)
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    # Join the processed words back into a sentence
    preprocessed_text = ' '.join(words)

    return preprocessed_text


def creating_similarity(df):
  data = df[df['user'] != 'group notification']
  data = data[data['message'] != '<Media omitted>\n']
  data['message_preprocessed'] = data['message'].apply(text_preprocessing)
  data = data[['user','message_preprocessed']]
  data_final = data.groupby('user')['message_preprocessed'].apply(sum).reset_index()
  tfidf_vectorizer = TfidfVectorizer()
  tfidf_matrix = tfidf_vectorizer.fit_transform(data_final['message_preprocessed'])
  cosine_sim = cosine_similarity(tfidf_matrix,tfidf_matrix)
  user_names = data_final['user'].tolist()
  similarity = pd.DataFrame(cosine_sim, index=user_names, columns=user_names)

  return similarity

def get_user_user_similarity(sim,selected_user):
  d = sim[selected_user].reset_index().sort_values(by = selected_user, ascending = False).iloc[1:]
  d[selected_user] = d[selected_user]*100
  d.rename(columns = {'index': 'User',selected_user:'Percentage Similarity'},inplace =True)

  return d