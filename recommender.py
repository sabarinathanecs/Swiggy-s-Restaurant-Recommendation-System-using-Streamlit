import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def get_recommendations(input_vector, encoded_df, top_n=5):
    similarities = cosine_similarity([input_vector], encoded_df.values)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    return top_indices

