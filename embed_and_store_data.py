"""
- Prepare the text to embed for each reccord of your dataset.
    - Create the reccord.
        - Clean the text.
        - Concatenate fields.
- Choose a Sentence Embedding Model.
- Embed the text generated in the previous step for each reccord.
- Store the embeddings in a vector database (i.e. elasticsearch).

- References
    - Elasticsearch
        - https://www.elastic.co/search-labs/tutorials/search-tutorial/full-text-search/create-index
        - https://elasticsearch-py.readthedocs.io/en/v8.11.1/interactive.html
        - https://github.com/elastic/elasticsearch-labs/blob/main/notebooks/search/00-quick-start.ipynb
        - https://medium.com/@teeppiphat/install-elasticsearch-docker-on-macos-m1-7dfbb8876b99
        - https://gist.github.com/benjamin-smith/78d330e08994fb5ce0de

        docker inspect container_id

"""
import pandas as pd
from pprint import pprint
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")
movies = pd.read_csv("imdb_top_1000_for_search.csv")

class Search:
    def __init__(self):
        self.es = Elasticsearch('http://localhost:9200')  # <-- connection options need to be added here
        client_info = self.es.info()
        print('Connected to Elasticsearch!')
        pprint(client_info.body)

mappings = {
    "properties": {
        "title_vector": {
            "type": "dense_vector",
            "dims": 384,
            "index": "true",
            "similarity": "cosine",
        }
    }
}


mapping = {
    "settings": {
        "number_of_shards": 2,
        "number_of_replicas": 1
    },
    "mappings": {
        "properties": {
            "Title": {
                "type": "text" # formerly "string"
            },
            "Released_Year": {
                "type": "text"
            },
            "Genre": {
                "type": "text"
            },
            "Rating": {
                "type": "float"
            },
            "Overview": {
                "type": "text"
            },
            "Director": {
                "type": "text"
            },
            "Star1": {
                "type": "text"
            },
            "Star2": {
                "type": "text"
            },
            "Star3": {
                "type": "text"
            },
            "Star4": {
                "type": "text"
            },
            "Embedding_Text": {
                "type": "text"
            },
            "Embedding_Vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": "true",
                "similarity": "cosine",
            },
        }
    }
}

esearch = Search()


response = esearch.es.indices.delete(
    index="movies_index", 
    ignore_unavailable=True
    )

response = esearch.es.indices.create(
    index="movies_index",
    body=mapping,
    ignore=400 # ignore 400 already exists code
    )

for index, movie in movies.iterrows():
    document = {}
    document["Title"] = movie["Series_Title"]
    document["Released_Year"] = movie["Released_Year"]
    document["Genre"] = movie["Genre"]
    document["Rating"] = movie["IMDB_Rating"]
    document["Overview"] = movie["Overview"]
    document["Director"] = movie["Director"]
    document["Star1"] = movie["Star1"]
    document["Star2"] = movie["Star2"]
    document["Star3"] = movie["Star3"]
    document["Star4"] = movie["Star4"]
    document["Embedding_Text"] = movie["Text"]
    document["Embedding_Vector"] = model.encode(movie["Text"]).tolist()

    esearch.es.index(index="movies_index", document=document)
