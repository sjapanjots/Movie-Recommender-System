import pandas as pd
import re
import nltk 
import logging 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Setup Logging 
logging.basicConfig(
    level = logging.INFO,
    format = '[%(asctime)s] %(levelname)s - %(message)s',
    handlers = [
        logging.FileHandler(filename="preprocess.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logging.info("Starting preprocessing...")

nltk.download("punkt")
nltk.download('punkt_tab')
nltk.download("stopwords")

#text cleaniing 
stop_words = set(stopwords.words("english"))


#Load sample dataset
try: 
    df = pd.read_csv("movies.csv")
    logging.info("Dataset Loaded successfully. Total rows: %d", len(df))
except Exception as e:
    logging.error("Failed to load dtaset: %s", str(e))
    raise e


def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

    # filter the required columns for the recommendations 
    required_columns = ["title", "genres", "overview" , "keywords"]

    df = df[required_columns]

    df = df.dropna().reset_index(drop=True)

    df['combined'] = df['genres'] + ' ' + df['kewords']

    logging.info("Cleaning text")
    df['cleaned_text'] = df['combined'].apply(preprocess_text)
    logging.info("Text cleaned")

    # Vectorization 
    logging.info(" Victorizing using TF-IDF...")
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
    logging.info("TF IDF matrix shape: %s", tfidf_matrix.shape)

    # Cosine Similarity for recommendations 
    logging.info("Calculating cosine similarity...")
    cosine_sim = cosine_similarity(tfidf_matrix , tfidf_matrix)
    logging.info("Cosine similarity matrix generated")

    # Save everything 
    joblib.dump(df , 'df_cleaned.pkl')
    joblib.dunp(tfidf_matrix , 'tfidf_matrix.pkl')
    joblib.dump(cousine_sim , 'cosine_sim.pkl')
    logging.info("Data Saved to disk")

    logging.info("Preprocessing complete")