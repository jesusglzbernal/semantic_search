"""
    Create an Streamlit app that does the following:

    - Reads an input from the user
    - Embeds the input
    - Search the vector DB for the entries closest to the user input
    - Outputs/displays the closest entries found
"""

import streamlit as st
import pandas as pd
from pprint import pprint
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer


@st.cache_data
def load_model(model):
    return SentenceTransformer(model)

class Search:
    def __init__(self):
        self.es = Elasticsearch('http://localhost:9200')  # <-- connection options need to be added here
        client_info = self.es.info()
        print('Connected to Elasticsearch!')
        pprint(client_info.body)

my_model = "all-MiniLM-L6-v2"
model = load_model(my_model)
esearch = Search()

st.title('IMDB Movie Semantic Search')

similarity_threshold = st.sidebar.slider("similarity", 0.1, 1.0, 0.6, 0.05)

movie_query = st.text_input("What kind of movie do you want to watch?")

#movie_embedding = model.encode(movie_query).tolist()


response = esearch.es.search(
    index="movies_index",
    knn={
        "field": "Embedding_Vector",
        "query_vector": model.encode(movie_query),
        "k": 10,
        "num_candidates": 100,
    },
)

st.write(response)
