from surprise import Dataset, SVD, Reader
import pandas as pd

# Make dataframes for ratings and movies
ratings_df = pd.read_csv(r'data\ratings.csv', usecols=[0, 1, 2], names=["userId", "movieId", "rating"])
movies_df = pd.read_csv(r"data\movies.csv")

# path to dataset file
file_path = r"data\ratings.csv"

# As we're loading a custom dataset, we need to define a reader. 
reader = Reader(line_format='user item rating', sep=',')

data = Dataset.load_from_file(file_path, reader=reader)

# Train on full dataset
trainset = data.build_full_trainset()

# We'll use the famous SVD algorithm.
algo = SVD()
algo.fit(trainset)


# TODO: given userId, recommend best movies
def make_recs(userId):
    print(algo.predict("1", "2"))