import pandas as pd
import ast
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
# movies = pd.read_csv('tmdb_5000_movies.csv')
# credits = pd.read_csv('tmdb_5000_credits.csv')

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# movies = movies.merge(credits, on='title', suffixes=('_movie', '_credit'))
print(movies.columns)

# Merge on title
movies = movies.merge(credits, on='title')

# Select required columns
movies = movies[['id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

movies.dropna(inplace=True)

# Convert text columns to lists
def convert(text):
    return [i['name'] for i in ast.literal_eval(text)]

def collapse(L):
    return [i.replace(" ", "") for i in L]

movies['keywords'] = movies['keywords'].apply(convert)
movies['genres'] = movies['genres'].apply(collapse)
movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords']

# Create a new DataFrame
new = movies[['id', 'title', 'tags']]
new.loc[:, 'tags'] = new['tags'].apply(lambda x: " ".join(x))

# Vectorize
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new['tags']).toarray()

# Compute similarity
similarity = cosine_similarity(vector)

# Save files
pickle.dump(new, open('movie_list.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print("✅ Pickle files created successfully!")

def recommend_movies(movie_title, n=5):
    """
    Recommend similar movies based on a given movie title
    
    Parameters:
    movie_title (str): Title of the movie to find recommendations for
    n (int): Number of recommendations to return
    
    Returns:
    list: List of recommended movie titles
    """
    try:
        # Load data
        movie_list = pickle.load(open('movie_list.pkl', 'rb'))
        similarity = pickle.load(open('similarity.pkl', 'rb'))
        
        # Check if movie exists in our dataset
        if movie_title not in movie_list['title'].values:
            return []
            
        # Find the index of the movie in our dataset
        idx = movie_list[movie_list['title'] == movie_title].index[0]
        
        # Get similarity scores for this movie with all others
        sim_scores = list(enumerate(similarity[idx]))
        
        # Sort movies based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N most similar movies
        recommended_movies = [movie_list.iloc[i[0]]['title'] for i in sim_scores[1:n+1]]
        
        return recommended_movies
    except Exception as e:
        print(f"Error: {e}")
        return []
