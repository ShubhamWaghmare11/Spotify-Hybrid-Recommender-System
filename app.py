import streamlit as st
from content_based_filtering import content_recommendation
from collaborative_filtering import collaborative_recommendation
from scipy.sparse import load_npz
import pandas as pd
from numpy import load
from hybrid_recommendations import HybridRecommenderSystem as hrs


# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Spotify Recommender System 🎶",
    page_icon="🎧",
    layout="centered",
    initial_sidebar_state="auto"
)

# ---------- STYLING ----------
st.markdown("""
    <style>
    .recommend-card {
        background-color: #1e1e1e;
        color: white;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    .recommend-card h4 {
        color: #ffb300;
        margin-bottom: 5px;
    }
    .recommend-card p {
        margin-top: 0;
        font-size: 15px;
        color: #dcdcdc;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff6f61;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)



# load the data
cleaned_data_path = "data/cleaned_data.csv"
songs_data = pd.read_csv(cleaned_data_path)

# load the transformed data
transformed_data_path = "data/transformed_data.npz"
transformed_data = load_npz(transformed_data_path)

# load the track ids
track_ids_path = "data/track_ids.npy"
track_ids = load(track_ids_path,allow_pickle=True)

# load the filtered songs data
filtered_data_path = "data/collab_filtered_data.csv"
filtered_data = pd.read_csv(filtered_data_path)

# load the interaction matrix
interaction_matrix_path = "data/interaction_matrix.npz"
interaction_matrix = load_npz(interaction_matrix_path)

# load the transformed hybrid data
transformed_hybrid_data_path = "data/transformed_hybrid_data.npz"
transformed_hybrid_data = load_npz(transformed_hybrid_data_path)


# ---------- HEADER ----------
st.title("🎧 MelodyMatch")
st.markdown("Find your next favorite track using **content-based song recommendations.**")

st.write("---")

# ---------- INPUTS ----------
col1, col2 = st.columns(2)
with col1:
    song_name = st.text_input("🎵 Enter Song Name", placeholder="e.g., lose yourself").lower().strip()
with col2:
    artist_name = st.text_input("🎤 Enter Artist Name", placeholder="e.g., Eminem").lower().strip()

k = st.slider("Select number of recommendations", 5, 20, 10, step=5)


filtering_type = st.selectbox("select the type of filtering:", ("Content-Based Filtering", "Collaborative Filtering", "Hybrid Filtering"),index=2)

if filtering_type == "Hybrid Filtering":
    diversity = st.slider(label="Diversity in Recommendation",
                          min_value = 1,
                          max_value = 10,
                          value = 5,
                          step = 1)
    content_based_weight = 1- (diversity/10)

if st.button("Get Recommendations"):
    if filtering_type == "Content-Based Filtering":

        with st.spinner("Analyzing musical vibes..."):
            recommendations = content_recommendation(song_name, artist_name, songs_data, transformed_data, k=k)

        if len(recommendations) == 0:
            st.warning("⚠️ Song not found. Please check the song name and artist name.")
        else:
            st.success(f"✅ Found {len(recommendations)-1} recommendations for **{song_name.title()}** by **{artist_name.title()}**.")
            st.write("---")
            

            for i, row in recommendations.iterrows():
                song = row['name'].title()
                artist = row['artist'].title()
                preview = row.get('spotify_preview_url', None)

                # Highlight top recommendation
                if i == 0:
                    st.markdown("### 🎶 Currently Playing")
                elif i == 1:
                    st.markdown("### ⏭️ Next Up")

                with st.container():
                    st.markdown(f"""
                        <div class="recommend-card">
                            <h4>{i+1}. {song}</h4>
                            <p>by <b>{artist}</b></p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if preview:
                        st.audio(preview)
                    st.write("")
    elif filtering_type == "Collaborative Filtering":
        with st.spinner("Finding similar listeners..."):
            recommendations = collaborative_recommendation(song_name, artist_name, track_ids, filtered_data, interaction_matrix, k=k)

            if len(recommendations) == 0:
                st.warning("⚠️ Song not found. Please check the song name and artist name.")
            else:
                st.success(f"✅ Found {len(recommendations)-1} recommendations for **{song_name.title()}** by **{artist_name.title()}**.")
                st.write("---")
                

                for i, row in recommendations.iterrows():
                    song = row['name'].title()
                    artist = row['artist'].title()
                    preview = row.get('spotify_preview_url', None)

                    # Highlight top recommendation
                    if i == 0:
                        st.markdown("### 🎶 Currently Playing")
                    elif i == 1:
                        st.markdown("### ⏭️ Next Up")

                    with st.container():
                        st.markdown(f"""
                            <div class="recommend-card">
                                <h4>{i+1}. {song}</h4>
                                <p>by <b>{artist}</b></p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        if preview:
                            st.audio(preview)
                        st.write("")

    else:
        if ((filtered_data['name'].str.lower() == song_name) & (filtered_data['artist'].str.lower() == artist_name)).any():
            with st.spinner("Harmonizing recommendations..."):
                hybrid_recommender = hrs(    
                    number_of_recommendations= k,
                    weight_content_based= content_based_weight
                )

                recommendations = hybrid_recommender.give_recommendation(song_name= song_name,
                                                        artist_name= artist_name,
                                                        songs_data= filtered_data,
                                                        transformed_matrix= transformed_hybrid_data,
                                                        track_ids= track_ids,
                                                        interaction_matrix= interaction_matrix)

            st.success(f"✅ Found {len(recommendations)-1} recommendations for **{song_name.title()}** by **{artist_name.title()}**.")
            st.write("---")
            

            for i, row in recommendations.iterrows():
                song = row['name'].title()
                artist = row['artist'].title()
                preview = row.get('spotify_preview_url', None)

                # Highlight top recommendation
                if i == 0:
                    st.markdown("### 🎶 Currently Playing")
                elif i == 1:
                    st.markdown("### ⏭️ Next Up")

                with st.container():
                    st.markdown(f"""
                        <div class="recommend-card">
                            <h4>{i+1}. {song}</h4>
                            <p>by <b>{artist}</b></p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if preview:
                        st.audio(preview)
                    st.write("")
        else:
            st.warning("⚠️ Song not found in the dataset. Please check the song name and artist name or go with the content based recommendation.")