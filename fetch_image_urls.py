import requests
import os
from dotenv import load_dotenv
import pandas as pd
from time import sleep

load_dotenv()

API_KEY = os.getenv('TMDB_API_KEY') 


def get_poster_url(title):
    
    # Ensure rate limit is not exceeded
    sleep(0.1)
    
    # Remove year
    split = title.split("(")
    split = split[:-1]
    QUERY = "(".join(split).strip()
    
    # Reformat 'The' if required
    if QUERY.endswith(", The"):
        QUERY = "The " + QUERY.split(", The")[0]

    # Reformat 'A' if required
    if QUERY.endswith(", A"):
        QUERY = "A " + QUERY.split(", A")[0]
    
    # Handle 'a.k.a'
    if "a.k.a." in QUERY:
        QUERY = QUERY.split("a.k.a.")[1].strip()

    
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={QUERY}"
    response = requests.get(url)
    data = response.json()

    if data["results"]:
        movie = data["results"][0]  # first match
        print("Found movie:", movie["title"], movie["release_date"])

        poster_path = movie.get("poster_path")
        if poster_path:
            # Build poster URL
            base_url = "https://image.tmdb.org/t/p/w500"
            poster_url = f"{base_url}{poster_path}"
            print("Found Poster URL:", poster_url)
            return poster_url
    
    print(f"Could not fetch poster url: {QUERY}")
    return None

movies_df = pd.read_csv(r"data\movies.csv")
movies_df['posterUrl'] = [
    get_poster_url(str(movie))
    for movie in movies_df['title']
]

movies_df.to_csv(r"data\movies_with_urls.csv", index=False)