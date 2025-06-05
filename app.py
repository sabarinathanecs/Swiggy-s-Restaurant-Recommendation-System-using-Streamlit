import streamlit as st
import pandas as pd
from preprocess import clean_data, encode_data
from recommender import get_recommendations
from utils import load_encoder, get_input_vector

@st.cache_data
def load_data():
    raw_data = clean_data('data/restaurants.csv')
    encoded = encode_data(raw_data)
    return raw_data, encoded

st.title("🍽️ Swiggy Restaurant Recommendation System")

raw_df, encoded_df = load_data()
encoder = load_encoder()

city = st.selectbox("Select City", raw_df['city'].dropna().unique())
cuisine = st.selectbox("Preferred Cuisine", raw_df['cuisine'].dropna().unique())
rating = st.slider("Minimum Rating", 1.0, 5.0, 3.5)
cost = st.number_input("Maximum Cost for Two", min_value=50, max_value=5000, value=300)
rating_count = st.slider("Minimum Rating Count", 0, 5000, 100)

# 'name' field is dummy to satisfy encoder input
name = "User"

input_vector = get_input_vector(encoder, name, city, cuisine, rating, rating_count, cost)
top_indices = get_recommendations(input_vector, encoded_df)

st.subheader("Top Recommended Restaurants")
for i in top_indices:
    rest = raw_df.iloc[i]
    st.markdown(f"**{rest['name']}** - {rest['city']}")
    st.markdown(f"⭐ Rating: {rest['rating']} ({rest['rating_count']} reviews)")
    st.markdown(f"🍴 Cuisine: {rest['cuisine']}")
    st.markdown(f"💰 Cost for Two: ₹{rest['cost']}")
    st.markdown("---")

