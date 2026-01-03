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

