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
        self.selected_features = selected_features  # Allow feature selection
        self.class_counts = Counter()
        self.feature_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.feature_values = defaultdict(set)
        self.total_samples = 0
        self.features = []
        self.vocabulary_size = 0  # To track unique words in titles
        
    def extract_movie_features(self, movie_data):
        """Extract features from movie data based on actual CSV structure"""
        features = {}
        
        # Extract basic fields from your CSV structure
        title = movie_data['Title']
        year = int(movie_data['Year'])
        genre = movie_data['Genre']
        director = movie_data['Director']
        production_company = movie_data['Production Company']
        runtime = int(movie_data['Runtime (min)'])
        country = movie_data['Country of Origin']
        language = movie_data['Original Language']
        
        # Title-based features (using sets to avoid counting same word multiple times)
        title_lower = title.lower()
        title_words = set(title_lower.split())  # Use set to get unique words only
        
        features['title_length'] = 'short' if len(title) <= 15 else 'medium' if len(title) <= 30 else 'long'
        features['word_count'] = 'few' if len(title.split()) <= 2 else 'medium' if len(title.split()) <= 4 else 'many'
        features['unique_word_count'] = 'few' if len(title_words) <= 2 else 'medium' if len(title_words) <= 4 else 'many'
        features['has_number'] = 'yes' if any(char.isdigit() for char in title) else 'no'
        features['has_colon'] = 'yes' if ':' in title else 'no'
        features['starts_with_the'] = 'yes' if title_lower.startswith('the ') else 'no'
        features['title_has_article'] = 'yes' if any(title_lower.startswith(article + ' ') for article in ['the', 'a', 'an']) else 'no'
        
        # Direct features from your CSV
        features['genre'] = genre
        features['director'] = director
        features['production_company'] = production_company
        features['country'] = country
        features['language'] = language
        
        # Year-based features
        features['year_category'] = '2010' if year == 2010 else '2011'
        features['is_2011'] = 'yes' if year == 2011 else 'no'
        
        # Runtime-based features
        features['runtime_category'] = 'short' if runtime <= 100 else 'medium' if runtime <= 130 else 'long'
        features['runtime_exact'] = 'very_short' if runtime < 90 else 'short' if runtime < 110 else 'medium' if runtime < 140 else 'long'
        features['long_movie'] = 'yes' if runtime > 120 else 'no'
        
        # Director-based features (you can customize this list based on your data)
        major_directors = ['Steven Spielberg', 'Christopher Nolan', 'Michael Bay', 'David Fincher', 
                          'Ridley Scott', 'Martin Scorsese', 'James Cameron', 'Quentin Tarantino']
        features['major_director'] = 'yes' if director in major_directors else 'no'
        
        # Production company features
        major_studios = ['Warner Bros.', 'Universal Pictures', 'Disney', 'Columbia Pictures', 
                        'Paramount Pictures', 'DreamWorks', '20th Century Fox', 'Sony Pictures']
        features['major_studio'] = 'yes' if any(studio in production_company for studio in major_studios) else 'no'
        
        # Language and country features
        features['english_language'] = 'yes' if language == 'English' else 'no'
        features['usa_origin'] = 'yes' if country == 'USA' else 'no'
        
        # Genre-based features
        action_genres = ['Action', 'Adventure', 'Thriller']
        features['action_genre'] = 'yes' if any(g in genre for g in action_genres) else 'no'
        
        # Franchise/sequel indicators
        sequel_words = ['2', 'ii', 'iii', 'part', 'rise', 'dark', 'returns', 'begins', 'dawn', 'evolution']
        features['likely_sequel'] = 'yes' if any(word in title_lower for word in sequel_words) else 'no'
        
        # Filter features if specific ones are selected
        if self.selected_features:
            features = {k: v for k, v in features.items() if k in self.selected_features}
        
        return features
    
    def calculate_vocabulary_size(self, movie_data_list):
        """Calculate vocabulary size using unique words across all titles"""
        all_words = set()
        for movie_data in movie_data_list:
            title_words = set(movie_data['Title'].lower().split())
            all_words.update(title_words)
        self.vocabulary_size = len(all_words)
        return self.vocabulary_size
    
    def fit(self, movie_data_list, labels):
        """Train the classifier"""
        # Reset counters
        self.class_counts = Counter()
        self.feature_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.feature_values = defaultdict(set)
        
        # Calculate vocabulary size
        self.calculate_vocabulary_size(movie_data_list)
        
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
            print(f"Vocabulary size: {self.vocabulary_size}")
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
        return prediction if prediction is not None else "Failure"
    
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
                print(f"{status} '{movie_data['Title']}' -> Predicted: {predicted_label}, Actual: {true_label}")

        accuracy = correct / total
        if verbose:
            print(f"\nAccuracy: {correct}/{total} = {accuracy:.3f} ({accuracy*100:.1f}%)")

            # Confusion matrix and classification report
            print("\nConfusion Matrix:")
            cm = confusion_matrix(test_labels, predictions, labels=["Success", "Failure"])
            print(pd.DataFrame(cm, index=["Actual Success", "Actual Failure"], columns=["Predicted Success", "Predicted Failure"]))

            print("\nClassification Report:")
            print(classification_report(test_labels, predictions, digits=3))

        return accuracy, predictions

def load_movie_data(filename):
    """Load movie data from CSV file with proper field handling"""
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
                
                # Ensure all required string fields exist
                required_fields = ['Title', 'Genre', 'Director', 'Production Company', 
                                 'Country of Origin', 'Original Language']
                for field in required_fields:
                    if field not in row or not row[field]:
                        row[field] = 'Unknown'
                
                movies.append(row)
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not process row {row}. Error: {e}")
                continue
    return movies

def calculate_success_labels(movies):
    """Calculate average revenue and assign success labels - skip if Success column exists"""
    # Check if Success column already exists
    if 'Success' in movies[0]:
        print("Success labels already exist in dataset")
        labels = []
        success_count = 0
        
        for movie in movies:
            success_value = int(movie['Success'])
            if success_value == 1:
                labels.append("Success")
                success_count += 1
            else:
                labels.append("Failure")
        
        print(f"Successful movies: {success_count}/{len(movies)} ({success_count/len(movies)*100:.1f}%)")
        print(f"Failed movies: {len(movies)-success_count}/{len(movies)} ({(len(movies)-success_count)/len(movies)*100:.1f}%)")
        
        # Calculate average revenue for reference
        revenues = [movie["Gross Revenue (million)"] for movie in movies]
        average_revenue = sum(revenues) / len(revenues)
        return labels, average_revenue
    
    else:
        # Original logic for calculating success based on average revenue
        revenues = [movie["Gross Revenue (million)"] for movie in movies]
        average_revenue = sum(revenues) / len(revenues)
        
        print(f"Average Revenue: ${average_revenue:.1f} million")
        
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
        
        print(f"Successful movies: {success_count}/{len(movies)} ({success_count/len(movies)*100:.1f}%)")
        print(f"Failed movies: {len(movies)-success_count}/{len(movies)} ({(len(movies)-success_count)/len(movies)*100:.1f}%)")
        
        return labels, average_revenue

def random_split(movies, labels, train_ratio=0.7, random_seed=42):
    """Randomly split data into training and test sets"""
    random.seed(random_seed)
    
    # Create indices and shuffle them
    indices = list(range(len(movies)))
    random.shuffle(indices)
    
    # Split indices
    split_point = int(train_ratio * len(movies))
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]
    
    # Create train and test sets
    train_movies = [movies[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_movies = [movies[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    return train_movies, train_labels, test_movies, test_labels

def find_best_features(movies, labels, all_features):
    """Find the best combination of features"""
    print("\n" + "="*60)
    print("FEATURE SELECTION ANALYSIS")
    print("="*60)
    
    best_accuracy = 0
    best_features = None
    best_feature_count = 0
    results = []
    
    # Test individual features
    print("\n1. INDIVIDUAL FEATURES:")
    print("-" * 40)
    for feature in all_features:
        train_movies, train_labels, test_movies, test_labels = random_split(movies, labels)
        classifier = NaiveBayesClassifier(use_logs=True, smoothing=True, selected_features=[feature])
        classifier.fit(train_movies, train_labels)
        accuracy, _ = classifier.evaluate(test_movies, test_labels)
        
        results.append((1, [feature], accuracy))
        print(f"{feature:<25}: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_features = [feature]
            best_feature_count = 1
    
    # Test pairs of features
    print("\n2. PAIRS OF FEATURES:")
    print("-" * 40)
    best_pairs = []
    for feature_pair in combinations(all_features, 2):
        train_movies, train_labels, test_movies, test_labels = random_split(movies, labels)
        classifier = NaiveBayesClassifier(use_logs=True, smoothing=True, selected_features=list(feature_pair))
        classifier.fit(train_movies, train_labels)
        accuracy, _ = classifier.evaluate(test_movies, test_labels)
        
        results.append((2, list(feature_pair), accuracy))
        best_pairs.append((feature_pair, accuracy))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_features = list(feature_pair)
            best_feature_count = 2
    
    # Show top 5 pairs
    best_pairs.sort(key=lambda x: x[1], reverse=True)
    for feature_pair, accuracy in best_pairs[:5]:
        print(f"{' + '.join(feature_pair):<40}: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Test triplets of features
    print("\n3. TRIPLETS OF FEATURES:")
    print("-" * 40)
    best_triplets = []
    for feature_triplet in combinations(all_features, 3):
        train_movies, train_labels, test_movies, test_labels = random_split(movies, labels)
        classifier = NaiveBayesClassifier(use_logs=True, smoothing=True, selected_features=list(feature_triplet))
        classifier.fit(train_movies, train_labels)
        accuracy, _ = classifier.evaluate(test_movies, test_labels)
        
        results.append((3, list(feature_triplet), accuracy))
        best_triplets.append((feature_triplet, accuracy))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_features = list(feature_triplet)
            best_feature_count = 3
    
    # Show top 5 triplets
    best_triplets.sort(key=lambda x: x[1], reverse=True)
    for feature_triplet, accuracy in best_triplets[:5]:
        print(f"{' + '.join(feature_triplet):<50}: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Test all features
    print("\n4. ALL FEATURES:")
    print("-" * 40)
    train_movies, train_labels, test_movies, test_labels = random_split(movies, labels)
    classifier = NaiveBayesClassifier(use_logs=True, smoothing=True, selected_features=all_features)
    classifier.fit(train_movies, train_labels)
    accuracy, _ = classifier.evaluate(test_movies, test_labels)
    
    results.append((len(all_features), all_features, accuracy))
    print(f"All features: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_features = all_features
        best_feature_count = len(all_features)
    
    print(f"\n" + "="*60)
    print(f"BEST FEATURE COMBINATION:")
    print(f"Features ({best_feature_count}): {best_features}")
    print(f"Accuracy: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)")
    print("="*60)
    
    return best_features, best_accuracy, results

def learning_curve_analysis(movies, labels, best_features):
    """Analyze how accuracy changes with training set size"""
    print("\n" + "="*60)
    print("LEARNING CURVE ANALYSIS")
    print("="*60)
    
    # Split data once for consistent test set
    train_movies, train_labels, test_movies, test_labels = random_split(movies, labels, train_ratio=0.8)
    
    training_sizes = []
    accuracies = []
    
    # Test different training set sizes
    min_size = max(5, len(train_movies) // 10)  # At least 5 samples
    max_size = len(train_movies)
    step_size = max(1, (max_size - min_size) // 10)
    
    for size in range(min_size, max_size + 1, step_size):
        if size > len(train_movies):
            size = len(train_movies)
        
        # Use first 'size' training examples
        subset_movies = train_movies[:size]
        subset_labels = train_labels[:size]
        
        # Train classifier
        classifier = NaiveBayesClassifier(use_logs=True, smoothing=True, selected_features=best_features)
        classifier.fit(subset_movies, subset_labels)
        
        # Evaluate
        accuracy, _ = classifier.evaluate(test_movies, test_labels)
        
        training_sizes.append(size)
        accuracies.append(accuracy)
        
        print(f"Training size: {size:2d}, Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Find optimal training size (where accuracy plateaus)
    max_accuracy = max(accuracies)
    optimal_size = None
    
    for i, (size, acc) in enumerate(zip(training_sizes, accuracies)):
        if acc >= max_accuracy * 0.95:  # Within 95% of max accuracy
            optimal_size = size
            break
    
    print(f"\nMaximum accuracy: {max_accuracy:.3f} ({max_accuracy*100:.1f}%)")
    print(f"Optimal training size: {optimal_size} examples")
    print(f"Training efficiency: {optimal_size}/{training_sizes[-1]} = {optimal_size/training_sizes[-1]*100:.1f}%")
    
    # Plot learning curve
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(training_sizes, [acc*100 for acc in accuracies], 'b-o', linewidth=2, markersize=6)
        plt.axhline(y=max_accuracy*100, color='r', linestyle='--', alpha=0.7, label=f'Max Accuracy ({max_accuracy*100:.1f}%)')
        plt.axvline(x=optimal_size, color='g', linestyle='--', alpha=0.7, label=f'Optimal Size ({optimal_size})')
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy (%)')
        plt.title('Learning Curve: Accuracy vs Training Set Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Learning curve saved as 'learning_curve.png'")
    except Exception as e:
        print(f"Could not create plot: {e}")
    
    return training_sizes, accuracies, optimal_size

def comprehensive_evaluation(movies, labels, best_features, num_trials=10):
    """Perform multiple random splits to get robust accuracy estimates"""
    print(f"\n" + "="*60)
    print(f"COMPREHENSIVE EVALUATION ({num_trials} trials)")
    print("="*60)
    
    accuracies = []
    
    for trial in range(num_trials):
        # Different random seed for each trial
        train_movies, train_labels, test_movies, test_labels = random_split(
            movies, labels, random_seed=42 + trial
        )
        
        classifier = NaiveBayesClassifier(use_logs=True, smoothing=True, selected_features=best_features)
        classifier.fit(train_movies, train_labels)
        accuracy, _ = classifier.evaluate(test_movies, test_labels)
        
        accuracies.append(accuracy)
        print(f"Trial {trial+1:2d}: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    print(f"\nResults Summary:")
    print(f"Mean Accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f} ({mean_accuracy*100:.1f}% ± {std_accuracy*100:.1f}%)")
    print(f"Best Trial: {max(accuracies):.3f} ({max(accuracies)*100:.1f}%)")
    print(f"Worst Trial: {min(accuracies):.3f} ({min(accuracies)*100:.1f}%)")
    
    return mean_accuracy, std_accuracy

def main():
    print("=== ENHANCED MOVIE SUCCESS PREDICTION SYSTEM ===")
    
    # Load data from CSV
    input_filename = '270movies_with_success.csv'
    
    try:
        movie_dataset = load_movie_data(input_filename)
        print(f"Loaded {len(movie_dataset)} movies from {input_filename}")
    except FileNotFoundError:
        print(f"Error: File '{input_filename}' not found. Please ensure it's in the same directory.")
        return
    
    # Calculate success labels
    labels, avg_revenue = calculate_success_labels(movie_dataset)
    
    # Get all available features
    sample_classifier = NaiveBayesClassifier()
    sample_features = sample_classifier.extract_movie_features(movie_dataset[0])
    all_features = list(sample_features.keys())
    
    print(f"\nAvailable features: {all_features}")
    print(f"Total features: {len(all_features)}")
    
    # Find best feature combination
    best_features, best_accuracy, feature_results = find_best_features(movie_dataset, labels, all_features)
    
    # Learning curve analysis
    training_sizes, accuracies, optimal_size = learning_curve_analysis(movie_dataset, labels, best_features)
    
    # Comprehensive evaluation
    mean_acc, std_acc = comprehensive_evaluation(movie_dataset, labels, best_features)
    
    # Final detailed evaluation with best configuration
    print(f"\n" + "="*60)
    print("FINAL DETAILED EVALUATION")
    print("="*60)
    
    train_movies, train_labels, test_movies, test_labels = random_split(movie_dataset, labels)
    
    final_classifier = NaiveBayesClassifier(use_logs=True, smoothing=True, selected_features=best_features)
    final_classifier.fit(train_movies, train_labels)
    
    print(f"Training set: {len(train_movies)} movies")
    print(f"Test set: {len(test_movies)} movies")
    print(f"Using features: {best_features}")
    print(f"Vocabulary size: {final_classifier.vocabulary_size} unique words")
    
    accuracy, predictions = final_classifier.evaluate(test_movies, test_labels, verbose=True)
    
    # Show detailed prediction for a few examples
    print(f"\nDETAILED PREDICTION EXAMPLES:")
    print("-" * 50)
    for i, movie in enumerate(test_movies[:3]):
        print(f"\nExample {i+1}:")
        final_classifier.predict_proba(movie, verbose=True)
    
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Best features: {best_features}")
    print(f"Best single trial accuracy: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)")
    print(f"Average accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"Optimal training size: {optimal_size} examples")
    print(f"Vocabulary size: {final_classifier.vocabulary_size} unique words")


if __name__ == "__main__":
    main()