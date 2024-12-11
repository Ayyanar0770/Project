import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample data for training the model
emotion_genre_data = {
    "happy": "Comedy",
    "sad": "Drama",
    "excited": "Action",
    "fear": "Horror",
    "love": "Romance",
    "curious": "Mystery",
    "inspired": "Biography",
    "adventurous": "Adventure",
    "tense": "Thriller",
    "intrigued": "Crime"
}

# Prepare training data
emotions = list(emotion_genre_data.keys())
genres = list(emotion_genre_data.values())

# Vectorize emotions
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(emotions)

# Encode genres
genre_encoder = {genre: i for i, genre in enumerate(set(genres))}
y = [genre_encoder[genre] for genre in genres]

# Train logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Save model and vectorizer
with open('emotion_genre_model.pkl', 'wb') as file:
    pickle.dump((vectorizer, model, genre_encoder), file)

import csv
from tabulate import tabulate
import pickle

# Dictionary to map emotions to genres
GENRES = {
    "Drama": 'Drama',
    "Action": 'Action',
    "Thriller": 'Thriller',
    "Adventure": 'Adventure',
    "Comedy": 'Comedy',
    "Biography": 'Biography',
    "Mystery": 'Mystery',
    "Romance": 'Romance',
    "Crime": 'Crime',
    "Horror": 'Horror',
}

def load_movies_from_csv(file_path):
    """Load movie data from a CSV file and return a list of dictionaries."""
    movies = []
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                movies.append(row)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"Error reading file: {e}")
    return movies

def filter_movies_by_genre(movies, genre):
    """Filter movies by the specified genre."""
    return [movie for movie in movies if genre in movie['Genre'].split(', ')]

def predict_genre(emotion):
    """Predict the genre based on the given emotion using the trained model."""
    with open('emotion_genre_model.pkl', 'rb') as file:
        vectorizer, model, genre_encoder = pickle.load(file)
    
    emotion_vector = vectorizer.transform([emotion])
    predicted_genre_index = model.predict(emotion_vector)[0]
    
    genre_decoder = {i: genre for genre, i in genre_encoder.items()}
    return genre_decoder.get(predicted_genre_index)

def main(emotion):
    genre = predict_genre(emotion)
    if not genre:
        print("Invalid emotion.")
        return []

    # Load movies from CSV
    movies = load_movies_from_csv('movies.csv')
    
    # Filter movies by the specified genre
    filtered_movies = filter_movies_by_genre(movies, genre)
    
    # Sort movies by ratings in descending order
    filtered_movies.sort(key=lambda x: float(x['Rating']), reverse=True)

    return filtered_movies

# Driver Function
if __name__ == '__main__':
    emotion = input("Enter the emotion: ").strip()
    movie_data = main(emotion)

    if not movie_data:
        print("No titles found.")
    else:
        max_titles = 50 #if emotion in GENRES else 50
        # Prepare data for tabulate
        table_data = [
            [
                index + 1,
                movie['Name'],
                movie['Year'],
                movie['Director'],
                movie['Actor'],
                movie['Rating'],
                movie['Genre'],
                movie['movie_rating'],
                movie['content_rating']
            ]
            for index, movie in enumerate(movie_data[:max_titles])
        ]
        # Define headers with added spacing
        headers = [
            "Sno", 
            "Title", 
            "Year", 
            "Director", 
            "Actor", 
            "Rating", 
            "Genre", 
            "MovieRating", 
            "ContentRating"
        ]

        # Set column alignment
        align = ["center"] * len(headers)

        # Print table with adjusted headers
        print(tabulate(table_data, headers=headers, tablefmt='grid', colalign=align))
