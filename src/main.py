# app.py

import json
import streamlit as st
from recommend import df, recommend_movies
from omdb_utils import get_movie_details

config = json.load(open('config.json'))

# OMDB API key 
OMDB_API_KEY = config['OMDB_API_KEY']

st.set_page_config(
    page_title="Movie Recommender System",
     page_icon="🎬",
      layout="wide"
      )

st.title("Movie Recommender 🎬")

# Using 'title' instead of a song now 
movie_list = sorted(df['title'].dropna().unique)
selected_movies = st.selecbox("🎬 Select a movie" , movie_list)

if st.button("🚀 Recommend Similar Movies"):
    with st.spinner("🔍 Generating recommendations..."):
        recommendations = recommend_movies(selected_movies)
        if recommendations is None or recommendations.empty:
            st.warning("No recommendations found for this movie.")
        else:
            st.success("Top similar movies")
            for _, row in recommendations.iterrows():
                movie_title = row['title']
                plot, poster = get_movie_details(movie_title, OMDB_API_KEY)

                with st.container():
                    col1,col2 = st.columns([1,3])
                    with col1:
                        if poster != "N/A":
                            st.image(poster , width= 100)

                        else: 
                            st.write("No poster available")

                    with col2:
                        st.markdown(f"### {movie_title}")
                        st.markdown(f"*{plot}*" if plot != "N/A" else "No plot available")