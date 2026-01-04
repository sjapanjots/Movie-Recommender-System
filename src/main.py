import os
import streamlit as st
from dotenv import load_dotenv
from recommend import df, recommend_movies
from omdb_utils import get_movie_details

# Load .env for local development
load_dotenv()

# Get API key (works locally + Streamlit Cloud)
OMDB_API_KEY = os.getenv("OMDB_API_KEY") or st.secrets.get("OMDB_API_KEY")

if not OMDB_API_KEY:
    st.error("OMDB API key not found. Please configure it in Streamlit Secrets.")
    st.stop()

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Recommender")

# Movie selection
movie_list = sorted(df['title'].dropna().unique())
selected_movie = st.selectbox("🎬 Select a movie:", movie_list)

if st.button("🚀 Recommend Similar Movies"):
    with st.spinner("Finding similar movies..."):
        recommendations = recommend_movies(selected_movie)

        if recommendations is None or recommendations.empty:
            st.warning("Sorry, no recommendations found.")
        else:
            st.success("Top similar movies:")

            for _, row in recommendations.iterrows():
                movie_title = row['title']
                plot, poster = get_movie_details(movie_title, OMDB_API_KEY)

                with st.container():
                    col1, col2 = st.columns([1, 3])

                    with col1:
                        if poster != "N/A":
                            st.image(poster, width=100)
                        else:
                            st.write("❌ No Poster Found")

                    with col2:
                        st.markdown(f"### {movie_title}")
                        st.markdown(
                            f"*{plot}*" if plot != "N/A" else "_Plot not available_"
                        )

# Footer
st.markdown(
    """
    <style>
    .footer {
        text-align: center;
        color: #888;
        font-size: 13px;
        margin-top: 40px;
    }
    </style>
    <div class="footer">
        🎬 Designed & Developed by <b>Japanjot Singh</b>
    </div>
    """,
    unsafe_allow_html=True
)
