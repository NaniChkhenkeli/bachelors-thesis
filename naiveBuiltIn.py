import csv
from collections import defaultdict
import math
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

class MovieClassifier:
    def __init__(self):
        self.model = CategoricalNB()
        self.preprocessor = None
        self.label_encoder = LabelEncoder()
        self.features = []
        
    def extract_movie_features(self, movie_data):
        """Extract features from movie data"""
        features = {}
        
        title = movie_data['Title']
        year = int(movie_data['Year'])
        genre = movie_data['Genre']
        director = movie_data['Director']
        runtime = int(movie_data['Runtime (min)'])
        
        # Title-based features
        title_lower = title.lower()
        features['title_length'] = 'short' if len(title) <= 15 else 'medium' if len(title) <= 30 else 'long'
        features['word_count'] = 'few' if len(title.split()) <= 2 else 'medium' if len(title.split()) <= 4 else 'many'
        features['has_number'] = 'yes' if any(char.isdigit() for char in title) else 'no'
        features['has_colon'] = 'yes' if ':' in title else 'no'
        features['starts_with_the'] = 'yes' if title_lower.startswith('the ') else 'no'
        
        # Movie metadata features
        features['genre'] = genre
        features['year_category'] = '2010' if year == 2010 else '2011'
        features['runtime_category'] = 'short' if runtime <= 100 else 'medium' if runtime <= 130 else 'long'
        
        # Director experience (based on provided data)
        experienced_directors = ['Steven Spielberg', 'Christopher Nolan', 'Michael Bay', 'David Fincher']
        features['experienced_director'] = 'yes' if director in experienced_directors else 'no'
        
        # Franchise/sequel indicators
        sequel_words = ['2', 'ii', 'part', 'rise', 'dark', 'breaking dawn']
        features['likely_sequel'] = 'yes' if any(word in title_lower for word in sequel_words) else 'no'
        
        return features
    
    def fit(self, movie_data_list, labels):
        """Train the classifier"""
        # Extract features
        features_list = [self.extract_movie_features(movie) for movie in movie_data_list]
        self.features = list(features_list[0].keys()) if features_list else []
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(features_list)
        
        # Encode labels
        y = self.label_encoder.fit_transform(labels)
        
        # Create preprocessor for one-hot encoding
        self.preprocessor = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), self.features)],
            remainder='passthrough'
        )
        
        # Preprocess features
        X = self.preprocessor.fit_transform(df)
        
        # Train the model
        self.model.fit(X, y)
    
    def predict(self, movie_data):
        """Predict class for a movie"""
        features = self.extract_movie_features(movie_data)
        df = pd.DataFrame([features])
        X = self.preprocessor.transform(df)
        prediction = self.model.predict(X)
        return self.label_encoder.inverse_transform(prediction)[0]
    
    def predict_proba(self, movie_data):
        """Predict class probabilities for a movie"""
        features = self.extract_movie_features(movie_data)
        df = pd.DataFrame([features])
        X = self.preprocessor.transform(df)
        proba = self.model.predict_proba(X)[0]
        return {cls: proba[i] for i, cls in enumerate(self.label_encoder.classes_)}
    
    def evaluate(self, test_movies, test_labels, verbose=False):
        """Evaluate classifier accuracy"""
        predictions = [self.predict(movie) for movie in test_movies]
        correct = sum(1 for pred, true in zip(predictions, test_labels) if pred == true)
        accuracy = correct / len(test_labels)
        
        if verbose:
            print("=== EVALUATION RESULTS ===")
            for i, (movie_data, true_label) in enumerate(zip(test_movies, test_labels)):
                status = "✓" if predictions[i] == true_label else "✗"
                print(f"{status} '{movie_data['Title']}' (${movie_data['Gross Revenue (million)']}M) -> Predicted: {predictions[i]}, Actual: {true_label}")
            print(f"\nAccuracy: {correct}/{len(test_labels)} = {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        return accuracy, predictions

def load_movie_data(filename):
    """Load movie data from CSV file"""
    movies = []
    with open(filename, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Convert numerical fields to appropriate types
            row['Year'] = int(row['Year'])
            row['Runtime (min)'] = int(row['Runtime (min)'])
            row['Gross Revenue (million)'] = float(row['Gross Revenue (million)'])
            movies.append(row)
    return movies

def calculate_success_labels(movies):
    """Calculate average revenue and assign success labels"""
    revenues = [movie["Gross Revenue (million)"] for movie in movies]
    average_revenue = sum(revenues) / len(revenues)
    
    print(f"Average Revenue: ${average_revenue:.1f} million")
    print("="*50)
    
    labels = []
    success_count = 0
    
    for movie in movies:
        revenue = movie["Gross Revenue (million)"]
        if revenue > average_revenue:
            labels.append("Success")
            success_count += 1
            status = "SUCCESS"
        else:
            labels.append("Failure")
            status = "FAILURE"
        
        print(f"{movie['Title']:<40} ${revenue:>6.1f}M -> {status}")
    
    print(f"\nSuccessful movies: {success_count}/{len(movies)} ({success_count/len(movies)*100:.1f}%)")
    print(f"Failed movies: {len(movies)-success_count}/{len(movies)} ({(len(movies)-success_count)/len(movies)*100:.1f}%)")
    
    return labels, average_revenue

def save_dataset_with_success(movies, labels, output_filename):
    """Save the dataset with success labels to a new CSV file"""
    fieldnames = list(movies[0].keys()) + ['Success']
    
    with open(output_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        for movie, label in zip(movies, labels):
            movie_copy = movie.copy()
            movie_copy['Success'] = 1 if label == "Success" else 0
            writer.writerow(movie_copy)
    
    print(f"\nDataset with success labels saved to {output_filename}")

def main():
    print("=== MOVIE SUCCESS PREDICTION USING CSV DATASET ===")
    
    # Load data from CSV
    input_filename = 'classifier/15movies.csv'
    output_filename = '15movies_with_success.csv'
    
    try:
        movie_dataset = load_movie_data(input_filename)
    except FileNotFoundError:
        print(f"Error: File '{input_filename}' not found. Please ensure it's in the same directory.")
        return
    
    # Calculate success labels
    labels, avg_revenue = calculate_success_labels(movie_dataset)
    
    # Save dataset with success labels
    save_dataset_with_success(movie_dataset, labels, output_filename)
    
    # Split data for training and testing
    split_point = int(0.75 * len(movie_dataset))
    train_movies = movie_dataset[:split_point]
    train_labels = labels[:split_point]
    test_movies = movie_dataset[split_point:]
    test_labels = labels[split_point:]
    
    print(f"\nTraining set: {len(train_movies)} movies")
    print(f"Test set: {len(test_movies)} movies")
    
    # Create and train classifier
    print(f"\n{'='*60}")
    print("USING SCIKIT-LEARN CATEGORICAL NAIVE BAYES")
    print('='*60)
    
    classifier = MovieClassifier()
    classifier.fit(train_movies, train_labels)
    
    # Evaluate
    accuracy, predictions = classifier.evaluate(test_movies, test_labels, verbose=True)
    
    # Show detailed prediction for one example
    if test_movies:
        example_movie = test_movies[0]
        print(f"\nDetailed prediction example:")
        proba = classifier.predict_proba(example_movie)
        print(f"Class probabilities: {proba}")
    
    # Feature analysis (scikit-learn doesn't provide the same detailed feature counts)
    print("\nNote: For detailed feature analysis, you may want to examine the feature importance")
    print("or use other interpretability tools from scikit-learn.")
    
    # Test on hypothetical new movies
    new_movies = [
        {"Title": "Avatar 3", "Year": 2011, "Genre": "Sci-Fi", "Director": "James Cameron", "Runtime (min)": 150},
        {"Title": "The Comedy Show", "Year": 2011, "Genre": "Comedy", "Director": "Unknown Director", "Runtime (min)": 95},
        {"Title": "Dark Knight Returns", "Year": 2011, "Genre": "Action", "Director": "Christopher Nolan", "Runtime (min)": 145},
    ]
    
    print(f"\nPredictions for hypothetical new movies:")
    print("-" * 50)
    
    for movie in new_movies:
        # Add missing fields for feature extraction
        movie.update({
            "Production Company": "Unknown",
            "Country of Origin": "USA", 
            "Original Language": "English",
            "Gross Revenue (million)": 0  # Not used for prediction
        })
        
        prediction = classifier.predict(movie)
        print(f"'{movie['Title']}' ({movie['Genre']}, {movie['Director']}) -> {prediction}")

if __name__ == "__main__":
    main()