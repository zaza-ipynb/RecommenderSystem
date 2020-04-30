import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("book_data.csv")
df = df.dropna()

df2 = df['genres'].str.split('|', expand=True)
df2.head(10)
df['genres'] = df2[0]

df.drop_duplicates('book_title', inplace=True)


# Function for recommending books based on Book title. It takes book title and genre as an input.
def recommend(title):
    genre = list((df.loc[df['book_title'] == title]['genres']).copy().head(1).to_string())
    result = []
    for idx, value in enumerate(genre):
        if not(str.isdigit(value) or value==' '):
            result = result + genre[idx:]
            break
    genre = ''.join(result)
    
    # Matching the genre with the dataset and reset the index
    data = df.loc[df['genres'] == genre].copy() 
    data.reset_index(level = 0, inplace = True) 

    # Convert the index into series
    indices = pd.Series(data.index, index = data['book_title'])
    
    #Converting the book title into vectors and used bigram
    tf = TfidfVectorizer(analyzer='word', ngram_range=(2, 2), min_df = 1, stop_words='english')
    tfidf_matrix = tf.fit_transform(data['book_title'])
    
    # Calculating the similarity measures based on Cosine Similarity
    sg = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    # Get the index corresponding to original_title
    idx = indices[title]
    # Get the pairwsie similarity scores 
    sig = list(enumerate(sg[idx]))
    # Sort the books
    sig = sorted(sig, key=lambda x: x[1], reverse=True)
    # Scores of the 5 most similar books 
    sig = sig[1:6]
    # Book indicies
    book_indices = [i[0] for i in sig]
   
    # Top 5 book recommendation
    rec = data[['book_title', 'image_url', 'genres']].iloc[book_indices]
    return rec


def titles():
    return df['book_title'].values.tolist()