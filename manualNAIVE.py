import csv
from collections import Counter, defaultdict
import math
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

class NaiveBayesClassifier:
    def __init__(self, use_logs=True, smoothing=True):
        self.use_logs = use_logs
        self.smoothing = smoothing
        self.class_counts = Counter()
        self.feature_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.feature_values = defaultdict(set)
        self.total_samples = 0
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
        # Reset counters
        self.class_counts = Counter()
        self.feature_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.feature_values = defaultdict(set)
        
        # Process training data
        training_data = []
        for movie_data, label in zip(movie_data_list, labels):
            features = self.extract_movie_features(movie_data)
            training_data.append((features, label))
        
        # Get feature names from first sample
        self.features = list(training_data[0][0].keys()) if training_data else []
        self.total_samples = len(training_data)
        
        # Count classes and feature values
        for features, label in training_data:
            self.class_counts[label] += 1
            for feature in self.features:
                value = features[feature]
                self.feature_values[feature].add(value)
                self.feature_counts[label][feature][value] += 1
    
    def predict_proba(self, movie_data, verbose=False):
        """Predict class probabilities for a movie"""
        features = self.extract_movie_features(movie_data)
        
        if verbose:
            print(f"\nPredicting for movie: '{movie_data['Title']}'")
            print(f"Extracted features: {features}")
            print("\nDetailed Computation:")
        
        best_class = None
        best_score = float('-inf') if self.use_logs else 0
        class_probabilities = {}
        
        for cls in self.class_counts:
            # Prior probability
            prior = self.class_counts[cls] / self.total_samples
            
            if self.use_logs:
                score = math.log(prior) if prior > 0 else float('-inf')
                if verbose:
                    print(f"\nFor class '{cls}':")
                    print(f"log P({cls}) = log({self.class_counts[cls]}/{self.total_samples}) = {score:.6f}")
            else:
                score = prior
                if verbose:
                    print(f"\nFor class '{cls}':")
                    print(f"P({cls}) = {self.class_counts[cls]}/{self.total_samples} = {score:.6f}")
            
            # Likelihood for each feature
            for feature in self.features:
                value = features[feature]
                count = self.feature_counts[cls][feature][value]
                total_in_class = self.class_counts[cls]
                
                if self.smoothing:
                    num_values = len(self.feature_values[feature])
                    smoothed_count = count + 1
                    smoothed_total = total_in_class + num_values
                    likelihood = smoothed_count / smoothed_total
                else:
                    likelihood = count / total_in_class if total_in_class > 0 else 0
                
                if self.use_logs:
                    if likelihood > 0:
                        score += math.log(likelihood)
                        if verbose:
                            if self.smoothing:
                                print(f"log P({value}|{cls}) = log(({count}+1)/({total_in_class}+{num_values})) = {math.log(likelihood):.6f}")
                            else:
                                print(f"log P({value}|{cls}) = log({count}/{total_in_class}) = {math.log(likelihood):.6f}")
                    else:
                        score = float('-inf')
                        if verbose:
                            print(f"log P({value}|{cls}) = log(0) = -inf (feature never seen in this class)")
                        break
                else:
                    score *= likelihood
                    if verbose:
                        if self.smoothing:
                            print(f"P({value}|{cls}) = ({count}+1)/({total_in_class}+{num_values}) = {likelihood:.6f}")
                        else:
                            print(f"P({value}|{cls}) = {count}/{total_in_class} = {likelihood:.6f}")
                    if score == 0:
                        break
            
            class_probabilities[cls] = score
            if verbose:
                if self.use_logs:
                    print(f"Total log probability = {score:.6f}")
                else:
                    print(f"Total probability = {score:.6f}")
            
            if score > best_score:
                best_score = score
                best_class = cls
        
        # If all classes resulted in -inf (no valid predictions), return the most common class
        if best_class is None and self.class_counts:
            best_class = self.class_counts.most_common(1)[0][0]
        
        return best_class, class_probabilities
    
    def predict(self, movie_data):
        """Predict class for a movie"""
        prediction, _ = self.predict_proba(movie_data)
        return prediction if prediction is not None else "Failure"  # Default to "Failure" if no prediction
    
    def evaluate(self, test_movies, test_labels, verbose=False):
        """Evaluate classifier with accuracy, confusion matrix, and precision/recall/F1"""
        correct = 0
        total = len(test_movies)
        predictions = []

        if verbose:
            print("=== EVALUATION RESULTS ===")

        for i, (movie_data, true_label) in enumerate(zip(test_movies, test_labels)):
            predicted_label = self.predict(movie_data)
            predictions.append(predicted_label)

            if predicted_label == true_label:
                correct += 1

            if verbose:
                status = "✓" if predicted_label == true_label else "✗"
                print(f"{status} '{movie_data['Title']}' (${movie_data['Gross Revenue (million)']}M) -> Predicted: {predicted_label}, Actual: {true_label}")

        accuracy = correct / total
        print(f"\nAccuracy: {correct}/{total} = {accuracy:.3f} ({accuracy*100:.1f}%)")

        # Confusion matrix and classification report
        print("\nConfusion Matrix:")
        cm = confusion_matrix(test_labels, predictions, labels=["Success", "Failure"])
        print(pd.DataFrame(cm, index=["Actual Success", "Actual Failure"], columns=["Predicted Success", "Predicted Failure"]))

        print("\nClassification Report:")
        print(classification_report(test_labels, predictions, digits=3))

        return accuracy, predictions


def load_movie_data(filename):
    """Load movie data from CSV file"""
    movies = []
    with open(filename, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                # Convert numerical fields to appropriate types
                row['Year'] = int(row['Year'])
                row['Runtime (min)'] = int(row['Runtime (min)'])
                
                # Handle missing/empty revenue values by setting to 0
                revenue = row['Gross Revenue (million)']
                row['Gross Revenue (million)'] = float(revenue) if revenue else 0.0
                
                movies.append(row)
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not process row {row}. Error: {e}")
                continue
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
    input_filename = 'classifier/dt300.csv'
    output_filename = '270movies_with_success.csv'
    
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
    
    # Test different configurations
    configs = [
        ("With Logs + Smoothing", True, True),
        ("With Logs, No Smoothing", True, False),
        ("No Logs + Smoothing", False, True),
        ("No Logs, No Smoothing", False, False),
    ]
    
    best_accuracy = 0
    best_config = None
    best_classifier = None
    
    for config_name, use_logs, smoothing in configs:
        print(f"\n{'='*60}")
        print(f"CONFIGURATION: {config_name}")
        print('='*60)
        
        # Create and train classifier
        classifier = NaiveBayesClassifier(use_logs=use_logs, smoothing=smoothing)
        classifier.fit(train_movies, train_labels)
        
        # Evaluate
        accuracy, predictions = classifier.evaluate(test_movies, test_labels, verbose=True)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_config = config_name
            best_classifier = classifier
        
        # Show detailed prediction for one example
        if test_movies:
            example_movie = test_movies[0]
            print(f"\nDetailed prediction example:")
            classifier.predict_proba(example_movie, verbose=True)
    
    print(f"\n{'='*60}")
    print(f"BEST CONFIGURATION: {best_config}")
    print(f"BEST ACCURACY: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)")
    print('='*60)
    
    # Feature analysis
    print(f"\nFEATURE ANALYSIS (using best classifier):")
    print("-" * 40)
    
    if best_classifier:
        print("Success vs Failure patterns:")
        for feature in best_classifier.features:
            print(f"\n{feature.upper()}:")
            for value in best_classifier.feature_values[feature]:
                success_count = best_classifier.feature_counts["Success"][feature][value]
                failure_count = best_classifier.feature_counts["Failure"][feature][value]
                total = success_count + failure_count
                if total > 0:
                    success_rate = success_count / total * 100
                    print(f"  {value}: {success_count}/{total} successful ({success_rate:.1f}%)")
    
    # Test on hypothetical new movies
    new_movies = [
        {"Title": "Avatar 3", "Year": 2011, "Genre": "Sci-Fi", "Director": "James Cameron", "Runtime (min)": 150},
        {"Title": "The Comedy Show", "Year": 2011, "Genre": "Comedy", "Director": "Unknown Director", "Runtime (min)": 95},
        {"Title": "Dark Knight Returns", "Year": 2011, "Genre": "Action", "Director": "Christopher Nolan", "Runtime (min)": 145},
    ]
    
    print(f"\nPredictions for hypothetical new movies:")
    print("-" * 50)
    
    if best_classifier:
        for movie in new_movies:
            # Add missing fields for feature extraction
            movie.update({
                "Production Company": "Unknown",
                "Country of Origin": "USA", 
                "Original Language": "English",
                "Gross Revenue (million)": 0  # Not used for prediction
            })
            
            prediction = best_classifier.predict(movie)
            print(f"'{movie['Title']}' ({movie['Genre']}, {movie['Director']}) -> {prediction}")


if __name__ == "__main__":
    main()