import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import pickle
import os

def clean_data(file_path):
    df = pd.read_csv(file_path)
    df.drop_duplicates(inplace=True)
    df.dropna(subset=['name', 'city', 'cuisine', 'rating', 'cost'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    os.makedirs("processed", exist_ok=True)
    df.to_csv('processed/cleaned_data.csv', index=False)
    return df

def encode_data(df):
    encoder = OneHotEncoder(sparse=False)
    encoded = encoder.fit_transform(df[['name', 'city', 'cuisine']])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['name', 'city', 'cuisine']))
    encoded_df[['rating', 'rating_count', 'cost']] = df[['rating', 'rating_count', 'cost']].reset_index(drop=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("processed", exist_ok=True)
    encoded_df.to_csv('processed/encoded_data.csv', index=False)
    with open('models/encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    return encoded_df

