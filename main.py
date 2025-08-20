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


# Make recommendations for the userId
def make_recs(userId, top_n=10):
    # Movies already rated by the user
    rated_by_user = set(ratings_df.loc[ratings_df['userId'] == userId, 'movieId'])

    # Filter out rated movies
    possible_recs = movies_df.loc[~movies_df['movieId'].isin(rated_by_user)].copy()

    # Predict ratings for unrated movies (list comprehension)
    possible_recs['estimate'] = [
        algo.predict(str(userId), str(movie_id)).est 
        for movie_id in possible_recs['movieId']
    ]

    # Select top N recommendations
    top_movies = possible_recs.nlargest(top_n, 'estimate')['title']

    print("Here are some recommendations:")
    print("\n".join(top_movies))

