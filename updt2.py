import csv
from collections import Counter, defaultdict
import math
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import random
import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np

class NaiveBayesClassifier:
    def __init__(self, use_logs=True, smoothing=True, selected_features=None):
        self.use_logs = use_logs
        self.smoothing = smoothing
        self.selected_features = selected_features
        self.class_counts = Counter()
        self.feature_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.feature_values = defaultdict(set)
        self.total_samples = 0
        self.features = []
        self.vocabulary_size = 0
        
    def extract_movie_features(self, movie_data):
        """Extract features from movie data"""
        features = {}
        
        title = movie_data['Title']
        year = int(movie_data['Year'])
        genre = movie_data['Genre']
        director = movie_data['Director']
        production_company = movie_data['Production Company']
        runtime = int(movie_data['Runtime (min)'])
        country = movie_data['Country of Origin']
        language = movie_data['Original Language']
        
        # Title-based features
        title_lower = title.lower()
        title_words = set(title_lower.split())
        
        features['title_length'] = 'short' if len(title) <= 15 else 'medium' if len(title) <= 30 else 'long'
        features['word_count'] = 'few' if len(title.split()) <= 2 else 'medium' if len(title.split()) <= 4 else 'many'
        features['unique_word_count'] = 'few' if len(title_words) <= 2 else 'medium' if len(title_words) <= 4 else 'many'
        features['has_number'] = 'yes' if any(char.isdigit() for char in title) else 'no'
        features['has_colon'] = 'yes' if ':' in title else 'no'
        features['starts_with_the'] = 'yes' if title_lower.startswith('the ') else 'no'
        features['title_has_article'] = 'yes' if any(title_lower.startswith(article + ' ') for article in ['the', 'a', 'an']) else 'no'
        
        # Metadata features
        features['genre'] = genre
        features['director'] = director
        features['production_company'] = production_company
        features['country'] = country
        features['language'] = language
        
        # Temporal features
        features['year_category'] = '2010' if year == 2010 else '2011'
        features['is_2011'] = 'yes' if year == 2011 else 'no'
        
        # Runtime features
        features['runtime_category'] = 'short' if runtime <= 100 else 'medium' if runtime <= 130 else 'long'
        features['runtime_exact'] = 'very_short' if runtime < 90 else 'short' if runtime < 110 else 'medium' if runtime < 140 else 'long'
        features['long_movie'] = 'yes' if runtime > 120 else 'no'
        
        # Professional features
        major_directors = ['Steven Spielberg', 'Christopher Nolan', 'Michael Bay', 'David Fincher']
        features['major_director'] = 'yes' if director in major_directors else 'no'
        
        major_studios = ['Warner Bros.', 'Universal Pictures', 'Disney', 'Columbia Pictures']
        features['major_studio'] = 'yes' if any(studio in production_company for studio in major_studios) else 'no'
        
        # Language features
        features['english_language'] = 'yes' if language == 'English' else 'no'
        features['usa_origin'] = 'yes' if country == 'USA' else 'no'
        
        # Genre features
        action_genres = ['Action', 'Adventure', 'Thriller']
        features['action_genre'] = 'yes' if any(g in genre for g in action_genres) else 'no'
        
        # Franchise features
        sequel_words = ['2', 'ii', 'iii', 'part', 'rise', 'dark', 'returns', 'begins']
        features['likely_sequel'] = 'yes' if any(word in title_lower for word in sequel_words) else 'no'
        
        if self.selected_features:
            features = {k: v for k, v in features.items() if k in self.selected_features}
        
        return features
    
    def calculate_vocabulary_size(self, movie_data_list):
        """Calculate unique words in all titles"""
        all_words = set()
        for movie_data in movie_data_list:
            title_words = set(movie_data['Title'].lower().split())
            all_words.update(title_words)
        self.vocabulary_size = len(all_words)
        return self.vocabulary_size
    
    def fit(self, movie_data_list, labels):
        """Train the classifier"""
        self.class_counts = Counter()
        self.feature_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.feature_values = defaultdict(set)
        
        self.calculate_vocabulary_size(movie_data_list)
        
        training_data = []
        for movie_data, label in zip(movie_data_list, labels):
            features = self.extract_movie_features(movie_data)
            training_data.append((features, label))
        
        self.features = list(training_data[0][0].keys()) if training_data else []
        self.total_samples = len(training_data)
        
        for features, label in training_data:
            self.class_counts[label] += 1
            for feature in self.features:
                value = features[feature]
                self.feature_values[feature].add(value)
                self.feature_counts[label][feature][value] += 1
    
    def predict_proba(self, movie_data, verbose=False):
        """Predict class probabilities"""
        features = self.extract_movie_features(movie_data)
        
        best_class = None
        best_score = float('-inf') if self.use_logs else 0
        class_probabilities = {}
        
        for cls in self.class_counts:
            prior = self.class_counts[cls] / self.total_samples
            
            if self.use_logs:
                score = math.log(prior) if prior > 0 else float('-inf')
            else:
                score = prior
            
            for feature in self.features:
                value = features[feature]
                count = self.feature_counts[cls][feature][value]
                total_in_class = self.class_counts[cls]
                
                if self.smoothing:
                    num_values = len(self.feature_values[feature])
                    likelihood = (count + 1) / (total_in_class + num_values)
                else:
                    likelihood = count / total_in_class if total_in_class > 0 else 0
                
                if self.use_logs:
                    if likelihood > 0:
                        score += math.log(likelihood)
                    else:
                        score = float('-inf')
                        break
                else:
                    score *= likelihood
                    if score == 0:
                        break
            
            class_probabilities[cls] = score
            
            if score > best_score:
                best_score = score
                best_class = cls
        
        if best_class is None and self.class_counts:
            best_class = self.class_counts.most_common(1)[0][0]
        
        return best_class, class_probabilities
    
    def predict(self, movie_data):
        """Predict class for a movie"""
        prediction, _ = self.predict_proba(movie_data)
        return prediction if prediction is not None else "Failure"
    
    def evaluate(self, test_movies, test_labels, verbose=False):
        """Evaluate classifier performance"""
        correct = 0
        total = len(test_movies)
        predictions = []

        for movie_data, true_label in zip(test_movies, test_labels):
            predicted_label = self.predict(movie_data)
            predictions.append(predicted_label)

            if predicted_label == true_label:
                correct += 1

        accuracy = correct / total
        
        if verbose:
            print(f"Accuracy: {correct}/{total} = {accuracy:.3f} ({accuracy*100:.1f}%)")
            cm = confusion_matrix(test_labels, predictions, labels=["Success", "Failure"])
            print("Confusion Matrix:")
            print(pd.DataFrame(cm, index=["Actual Success", "Actual Failure"], 
                             columns=["Predicted Success", "Predicted Failure"]))
            print("\nClassification Report:")
            print(classification_report(test_labels, predictions, digits=3))

        return accuracy, predictions


def load_movie_data(filename):
    """Load movie data from CSV"""
    movies = []
    with open(filename, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                row['Year'] = int(row['Year'])
                row['Runtime (min)'] = int(row['Runtime (min)'])
                revenue = row['Gross Revenue (million)']
                row['Gross Revenue (million)'] = float(revenue) if revenue else 0.0
                
                required_fields = ['Title', 'Genre', 'Director', 'Production Company']
                for field in required_fields:
                    if field not in row or not row[field]:
                        row[field] = 'Unknown'
                
                movies.append(row)
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not process row {row}. Error: {e}")
                continue
    return movies

def calculate_success_labels(movies):
    """Calculate success labels"""
    if 'Success' in movies[0]:
        labels = []
        success_count = 0
        for movie in movies:
            success_value = int(movie['Success'])
            if success_value == 1:
                labels.append("Success")
                success_count += 1
            else:
                labels.append("Failure")
        
        revenues = [movie["Gross Revenue (million)"] for movie in movies]
        average_revenue = sum(revenues) / len(revenues)
        
        print(f"Using existing Success labels")
        print(f"Successful movies: {success_count}/{len(movies)} ({success_count/len(movies)*100:.1f}%)")
        return labels, average_revenue
    
    else:
        revenues = [movie["Gross Revenue (million)"] for movie in movies]
        average_revenue = sum(revenues) / len(revenues)
        
        labels = []
        success_count = 0
        
        for movie in movies:
            revenue = movie["Gross Revenue (million)"]
            if revenue > average_revenue:
                labels.append("Success")
                success_count += 1
            else:
                labels.append("Failure")
        
        print(f"Average Revenue: ${average_revenue:.1f} million")
        print(f"Successful movies: {success_count}/{len(movies)} ({success_count/len(movies)*100:.1f}%)")
        return labels, average_revenue

def random_split(movies, labels, train_ratio=0.7, random_seed=42):
    """Split data into training and test sets"""
    random.seed(random_seed)
    indices = list(range(len(movies)))
    random.shuffle(indices)
    
    split_point = int(train_ratio * len(movies))
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]
    
    return ([movies[i] for i in train_indices],
            [labels[i] for i in train_indices],
            [movies[i] for i in test_indices],
            [labels[i] for i in test_indices])

def find_best_features(movies, labels, all_features, num_trials=5):
    """Find best feature combination"""
    print("\n" + "="*60)
    print("FEATURE SELECTION ANALYSIS")
    print("="*60)
    
    best_accuracy = 0
    best_features = None
    results = []
    
    # Test individual features
    print("\n1. INDIVIDUAL FEATURES:")
    print("-" * 40)
    for feature in all_features:
        accuracies = []
        for _ in range(num_trials):
            train_movies, train_labels, test_movies, test_labels = random_split(movies, labels)
            classifier = NaiveBayesClassifier(selected_features=[feature])
            classifier.fit(train_movies, train_labels)
            accuracy, _ = classifier.evaluate(test_movies, test_labels)
            accuracies.append(accuracy)
        
        avg_accuracy = np.mean(accuracies)
        results.append(([feature], avg_accuracy))
        print(f"{feature:<25}: {avg_accuracy:.3f} ({avg_accuracy*100:.1f}%)")
        
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_features = [feature]
    
    # Test pairs of top features
    print("\n2. FEATURE PAIRS:")
    print("-" * 40)
    top_features = [f[0][0] for f in sorted(results, key=lambda x: x[1], reverse=True)[:5]]  # Get top 5 single features
    
    for f1, f2 in combinations(top_features, 2):
        accuracies = []
        for _ in range(num_trials):
            train_movies, train_labels, test_movies, test_labels = random_split(movies, labels)
            classifier = NaiveBayesClassifier(selected_features=[f1, f2])
            classifier.fit(train_movies, train_labels)
            accuracy, _ = classifier.evaluate(test_movies, test_labels)
            accuracies.append(accuracy)
        
        avg_accuracy = np.mean(accuracies)
        results.append(([f1, f2], avg_accuracy))
        print(f"{f1} + {f2:<30}: {avg_accuracy:.3f} ({avg_accuracy*100:.1f}%)")
        
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_features = [f1, f2]
    
    print("\n" + "="*60)
    print(f"BEST FEATURE COMBINATION: {', '.join(best_features)}")
    print(f"ACCURACY: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)")
    print("="*60)
    
    return best_features, best_accuracy, results


def learning_curve_analysis(movies, labels, best_features):
    """Analyze learning curve"""
    print("\n" + "="*60)
    print("LEARNING CURVE ANALYSIS")
    print("="*60)
    
    train_movies, train_labels, test_movies, test_labels = random_split(movies, labels, train_ratio=0.8)
    
    training_sizes = []
    accuracies = []
    
    min_size = max(5, len(train_movies) // 5)
    step_size = max(1, (len(train_movies) - min_size) // 10)
    
    for size in range(min_size, len(train_movies) + 1, step_size):
        acc = []
        for _ in range(3):  # Multiple trials per size
            subset = random.sample(list(zip(train_movies, train_labels)), size)
            sub_movies, sub_labels = zip(*subset)
            
            classifier = NaiveBayesClassifier(selected_features=best_features)
            classifier.fit(sub_movies, sub_labels)
            accuracy, _ = classifier.evaluate(test_movies, test_labels)
            acc.append(accuracy)
        
        avg_acc = np.mean(acc)
        training_sizes.append(size)
        accuracies.append(avg_acc)
        print(f"Training size: {size:2d}, Accuracy: {avg_acc:.3f} ({avg_acc*100:.1f}%)")
    
    max_acc = max(accuracies)
    optimal_size = next(size for size, acc in zip(training_sizes, accuracies) if acc >= max_acc * 0.95)
    
    print(f"\nMaximum accuracy: {max_acc:.3f} ({max_acc*100:.1f}%)")
    print(f"Optimal training size: {optimal_size} examples")
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(training_sizes, [a * 100 for a in accuracies], 'bo-')
    plt.axhline(y=max_acc * 100, color='r', linestyle='--', label=f'Max Accuracy ({max_acc*100:.1f}%)')
    plt.axvline(x=optimal_size, color='g', linestyle='--', label=f'Optimal Size ({optimal_size})')
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy (%)')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Learning curve saved as 'learning_curve.png'")
    
    return training_sizes, accuracies, optimal_size

def comprehensive_evaluation(movies, labels, best_features, num_trials=10):
    """Run multiple evaluation trials"""
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION")
    print("="*60)
    
    accuracies = []
    
    for trial in range(num_trials):
        train_movies, train_labels, test_movies, test_labels = random_split(movies, labels, random_seed=42 + trial)
        classifier = NaiveBayesClassifier(selected_features=best_features)
        classifier.fit(train_movies, train_labels)
        accuracy, _ = classifier.evaluate(test_movies, test_labels)
        accuracies.append(accuracy)
        print(f"Trial {trial+1:2d}: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    print(f"\nMean Accuracy: {mean_acc:.3f} ± {std_acc:.3f} ({mean_acc*100:.1f}% ± {std_acc*100:.1f}%)")
    print(f"Best Trial: {max(accuracies):.3f} ({max(accuracies)*100:.1f}%)")
    print(f"Worst Trial: {min(accuracies):.3f} ({min(accuracies)*100:.1f}%)")
    
    return mean_acc, std_acc

def main():
    print("=== ENHANCED MOVIE SUCCESS PREDICTION SYSTEM ===")
    
    # Load data
    input_filename = '15movies_with_success.csv'
    try:
        movies = load_movie_data(input_filename)
        print(f"Loaded {len(movies)} movies from {input_filename}")
    except FileNotFoundError:
        print(f"Error: File '{input_filename}' not found.")
        return
    
    # Calculate labels
    labels, avg_revenue = calculate_success_labels(movies)
    
    # Get all features
    sample_features = NaiveBayesClassifier().extract_movie_features(movies[0])
    all_features = list(sample_features.keys())
    print(f"\nAvailable features ({len(all_features)}): {all_features}")
    
    # Feature selection
    best_features, best_acc, _ = find_best_features(movies, labels, all_features)
    
    # Learning curve
    learning_curve_analysis(movies, labels, best_features)
    
    # Comprehensive evaluation
    mean_acc, std_acc = comprehensive_evaluation(movies, labels, best_features)
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    train_movies, train_labels, test_movies, test_labels = random_split(movies, labels)
    classifier = NaiveBayesClassifier(selected_features=best_features)
    classifier.fit(train_movies, train_labels)
    
    print(f"Training set: {len(train_movies)} movies")
    print(f"Test set: {len(test_movies)} movies")
    print(f"Features: {best_features}")
    
    accuracy, _ = classifier.evaluate(test_movies, test_labels, verbose=True)
    
    # Show predictions for first 3 test movies
    print("\nPREDICTION EXAMPLES:")
    for i, movie in enumerate(test_movies[:3]):
        print(f"\nMovie: {movie['Title']}")
        pred, probs = classifier.predict_proba(movie, verbose=True)
        print(f"Final Prediction: {pred} (Probabilities: {probs})")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Best Features: {best_features}")
    print(f"Best Accuracy: {best_acc:.3f} ({best_acc*100:.1f}%)")
    print(f"Mean Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"Vocabulary Size: {classifier.vocabulary_size} unique words")

if __name__ == "__main__":
    main()