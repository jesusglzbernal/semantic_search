import kaggle
import pandas as pd
import zipfile

kaggle.api.authenticate()
dataset = "harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows"
kaggle.api.dataset_download_files(dataset)

with zipfile.ZipFile(
    "imdb-dataset-of-top-1000-movies-and-tv-shows.zip", "r"
) as zip_ref:
    zip_ref.extractall(".")

movies = pd.read_csv("imdb_top_1000.csv")
movies["Text"] = movies["Series_Title"] + " " + movies["Genre"] + " " + movies["Overview"] + " " + movies["Director"] + " " + movies["Star1"] + " " + movies["Star2"] + " " + movies["Star3"] + " " + movies["Star4"]
print(movies.columns)
print(movies[["Series_Title", "Overview"]].head(10))

#movies.to_csv("imdb_top_1000_for_search.csv", index=False)

print(movies.describe())
