from surprise import Dataset, SVD, Reader
import pandas as pd
import pickle


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

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(algo, f)

