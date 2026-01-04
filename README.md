# 🎬 Movie Recommender System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)
[![Development](https://img.shields.io/badge/Status-Development-yellow.svg)](https://github.com/sjapanjots/Movie-Recommender-System)

A machine learning-powered movie recommendation system that suggests personalized movies based on content-based filtering techniques. This project uses the TMDB 5000 Movie Dataset and provides an interactive web interface built with Streamlit.

> **⚠️ Note:** This is a development project intended for local use and learning purposes. It is **not production-ready** and should be run locally on your machine. For production deployment, additional considerations like security, scalability, error handling, and optimization would be required.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Dataset](#dataset)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

This Movie Recommender System uses **content-based filtering** to suggest movies similar to a user's preferences. By analyzing movie metadata such as genres, keywords, cast, crew, and plot descriptions, the system calculates similarity scores between movies and recommends those that are most similar to the user's selected movie.

### Key Highlights

- **Content-Based Filtering**: Recommends movies based on item features and similarity
- **Interactive UI**: User-friendly Streamlit web interface
- **Real-time Recommendations**: Instant movie suggestions with poster images
- **TMDB Integration**: Fetches movie posters and additional details from TMDB API
- **Extensive Dataset**: Trained on 5000+ movies from TMDB

## ✨ Features

- 🎥 **Movie Selection**: Choose from thousands of movies in the database
- 🔍 **Smart Recommendations**: Get top 5 similar movie suggestions
- 🖼️ **Visual Display**: See movie posters alongside recommendations
- ⚡ **Fast Performance**: Pre-computed similarity matrix for instant results
- 📊 **Rich Metadata**: Utilizes genres, keywords, cast, crew, and overview
- 🎨 **Clean Interface**: Intuitive and responsive web design

## 🖥️ Local Development

**This project is designed for local development and educational purposes only.**

### Why This Is Not Production-Ready

This project serves as a learning tool and proof-of-concept. To be production-ready, it would require:

- **Security Enhancements**: API key management using environment variables and secrets management
- **Error Handling**: Comprehensive exception handling and fallback mechanisms
- **Performance Optimization**: Caching strategies, database integration, and optimized similarity computations
- **Scalability**: Distributed computing for larger datasets, load balancing
- **Testing**: Unit tests, integration tests, and end-to-end testing
- **Monitoring**: Logging, analytics, and error tracking
- **User Management**: Authentication, user profiles, and personalized recommendations
- **Code Quality**: Code reviews, linting, type hints, and documentation
- **Deployment Infrastructure**: CI/CD pipelines, containerization (Docker), cloud hosting
- **Legal Compliance**: Terms of service, privacy policy, GDPR compliance

### Running Locally

This application runs entirely on your local machine:

```bash
# Quick start to see it in action
streamlit run app.py
```

## 🛠️ Technology Stack

### Programming Language
- **Python 3.8+**

### Libraries & Frameworks
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning (CountVectorizer, Cosine Similarity)
- **NLTK**: Natural language processing (stemming)
- **Requests**: API calls to TMDB
- **Pickle**: Model serialization

### Machine Learning Techniques
- Content-Based Filtering
- Cosine Similarity
- Text Vectorization (Bag of Words)
- Feature Engineering

## 📥 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/sjapanjots/Movie-Recommender-System.git
cd Movie-Recommender-System
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies

```bash
pip install -r requirements.txt
```

**Required Libraries:**
```
streamlit
pandas
numpy
scikit-learn
nltk
requests
pickle-mixin
```

### Step 4: Download NLTK Data (First time only)

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Step 5: Get TMDB API Key

1. Create a free account at [TMDB](https://www.themoviedb.org/)
2. Navigate to Settings → API
3. Request an API key (choose "Developer" option)
4. Copy your API key
5. Replace `YOUR_API_KEY` in `app.py` with your actual API key:

```python
api_key = "your_actual_api_key_here"
```

**Note:** For this local project, the API key is hardcoded in the source file. In a production environment, use environment variables or a secrets management system instead.

## 🚀 Usage

### Running the Application

1. Navigate to the project directory:
```bash
cd Movie-Recommender-System
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. The application will open in your default web browser at `http://localhost:8501`

### Using the Recommender

1. **Select a Movie**: Use the dropdown menu to choose a movie from the database
2. **Get Recommendations**: Click the "Show Recommendations" button
3. **View Results**: See 5 similar movies with their posters
4. **Explore More**: Select different movies to get new recommendations

### Training the Model (Optional)

If you want to retrain the model or modify the recommendation algorithm:

1. Open the Jupyter notebook:
```bash
jupyter notebook movie-recommender-system.ipynb
```

2. Run all cells to:
   - Load and preprocess the dataset
   - Create feature vectors
   - Compute similarity matrix
   - Save the model as pickle files

## 📁 Project Structure

```
Movie-Recommender-System/
│
├── app.py                          # Main Streamlit application
├── movie-recommender-system.ipynb  # Jupyter notebook for model training
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore file
│
├── data/                           # Dataset directory
│   ├── tmdb_5000_movies.csv       # Movies dataset
│   └── tmdb_5000_credits.csv      # Credits dataset
│
├── models/                         # Saved models
│   ├── movies.pkl                 # Processed movie data
│   ├── movie_dict.pkl             # Movie dictionary
│   └── similarity.pkl             # Similarity matrix
│
└── LICENSE                         # License file
```

## 🧠 How It Works

### 1. Data Preprocessing

- Merges movie and credits datasets
- Extracts relevant features: genres, keywords, cast, crew, overview
- Cleans and processes text data (lowercasing, removing spaces, stemming)

### 2. Feature Engineering

- Combines multiple features into a single "tags" column
- Creates metadata tags for each movie

### 3. Text Vectorization

- Uses **CountVectorizer** to convert text to numerical vectors
- Implements **Bag of Words** model with 5000 most common words
- Applies **Porter Stemmer** for word normalization

### 4. Similarity Calculation

- Computes **Cosine Similarity** between all movie vectors
- Creates a similarity matrix (5000 x 5000)
- Higher similarity score = more similar movies

### 5. Recommendation Generation

```python
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return movies_list
```

### Content-Based Filtering Algorithm

The system uses content-based filtering which:
- Analyzes item (movie) features
- Recommends items similar to what the user liked
- Based on movie metadata, not user ratings
- Independent of other users' preferences

**Example:**
If you like *Iron Man* (Action, Sci-Fi, Robert Downey Jr.), the system recommends similar movies like *The Avengers*, *Captain America*, etc.

## 📊 Dataset

### Source
- **TMDB 5000 Movie Dataset** from [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

### Dataset Details

**tmdb_5000_movies.csv** contains:
- Movie ID, Title, Overview
- Genres, Keywords
- Release Date, Runtime
- Budget, Revenue, Vote Average

**tmdb_5000_credits.csv** contains:
- Movie ID
- Cast (actors)
- Crew (director, producer, etc.)

### Sample Features Used

```
Tags = Keywords + Genres + Cast (top 3) + Crew (director) + Overview
```

## 🎓 Learning Outcomes

This project demonstrates:

- **Content-Based Recommendation Systems**: Understanding how to build recommender systems using item features
- **Natural Language Processing**: Text preprocessing, vectorization, and stemming
- **Machine Learning**: Cosine similarity, feature engineering
- **Web Development**: Building interactive applications with Streamlit
- **Data Processing**: Working with large datasets using Pandas
- **API Integration**: Fetching external data from REST APIs
- **Model Serialization**: Saving and loading ML models with Pickle

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the Project**
```bash
git fork https://github.com/sjapanjots/Movie-Recommender-System.git
```

2. **Create your Feature Branch**
```bash
git checkout -b feature/AmazingFeature
```

3. **Commit your Changes**
```bash
git commit -m 'Add some AmazingFeature'
```

4. **Push to the Branch**
```bash
git push origin feature/AmazingFeature
```

5. **Open a Pull Request**

### Areas for Improvement

- Add collaborative filtering alongside content-based filtering
- Implement hybrid recommendation system
- Add user authentication and rating functionality
- Include movie trailers and reviews integration
- Implement environment variables for API key management
- Add comprehensive error handling and logging
- Create unit and integration tests
- Add caching for better performance
- Implement a proper database (PostgreSQL, MongoDB)
- Add filtering options (year, genre, rating, language)
- Optimize for production deployment (Docker, cloud hosting)
- Add CI/CD pipeline
- Implement proper security measures

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Your Name** - [@sjapanjots](https://github.com/sjapanjots)

Project Link: [https://github.com/sjapanjots/Movie-Recommender-System](https://github.com/sjapanjots/Movie-Recommender-System)

## ⚠️ Disclaimer

This project is intended for **educational and development purposes only**. It demonstrates fundamental concepts of recommendation systems and is not designed for production use. The application runs locally and has not been tested or optimized for deployment in a production environment.

## 🙏 Acknowledgments

- [TMDB](https://www.themoviedb.org/) for the movie database and API
- [Kaggle](https://www.kaggle.com/) for providing the dataset
- [Streamlit](https://streamlit.io/) for the amazing web framework
- All contributors and supporters of this project

## 📚 References

- [Content-Based Filtering - Google Developers](https://developers.google.com/machine-learning/recommendation/content-based/basics)
- [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
- [Recommender Systems](https://en.wikipedia.org/wiki/Recommender_system)

---

**Made with ❤️ by [sjapanjots](https://github.com/sjapanjots)**

⭐ Star this repo if you found it helpful!