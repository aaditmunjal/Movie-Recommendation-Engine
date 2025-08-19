from surprise import accuracy, Dataset, SVD, Reader
from surprise.model_selection import train_test_split


# path to dataset file
file_path = r"data\ratings.csv"

# As we're loading a custom dataset, we need to define a reader. 
reader = Reader(line_format='user item rating', sep=',')

data = Dataset.load_from_file(file_path, reader=reader)

trainset, testset = train_test_split(data, test_size=0.25)

# We'll use the famous SVD algorithm.
algo = SVD()

algo.fit(trainset)

print(algo.estimate(1, 2))