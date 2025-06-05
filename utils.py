import pickle
import pandas as pd

def load_encoder():
    with open('models/encoder.pkl', 'rb') as f:
        return pickle.load(f)

def get_input_vector(encoder, name, city, cuisine, rating, rating_count, cost):
    input_df = pd.DataFrame([[name, city, cuisine]], columns=['name', 'city', 'cuisine'])
    encoded_cat = encoder.transform(input_df)
    numerical = [[rating, rating_count, cost]]
    return list(encoded_cat[0]) + numerical[0]

