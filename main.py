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
def make_recs(userId):
    # Get movies that have already been rated by the user
    ratedByUser = ratings_df[ratings_df['userId'] == userId]['movieId'].to_list()
    
    # Get movies that have not been rated by the user yet
    possibleRecs = movies_df[~movies_df['movieId'].isin(ratedByUser)].reset_index(drop=True)
    # Get a list of the movieIds of unrated movies
    recList = possibleRecs['movieId'].to_list()
    
    # Predict ratings for all unrated movies
    out = []
    for i in recList:
        out.append((algo.predict(str(userId), str(i))).est)

    possibleRecs['estimate'] = out

    # Recommend the top 10
    top_10 = possibleRecs.nlargest(10, 'estimate')['title'].to_list()
    
    print("Here are some recommendations: ")
    for i in top_10:
        print(i)

