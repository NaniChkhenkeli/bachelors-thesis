import csv
from collections import Counter, defaultdict
import math
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import random
import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
import os

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
        
        # Basic features
        title = movie_data['Title']
        features['title_length'] = 'short' if len(title) <= 15 else 'medium' if len(title) <= 30 else 'long'
        features['has_colon'] = 'yes' if ':' in title else 'no'
        
        # Director features
        director = movie_data['Director']
        features['director'] = director
        features['major_director'] = 'yes' if director in [
            'Steven Spielberg', 'Christopher Nolan', 'James Cameron'
        ] else 'no'
        
        # Genre features
        genre = movie_data['Genre']
        features['action_genre'] = 'yes' if 'Action' in genre else 'no'
        
        # Production features
        features['major_studio'] = 'yes' if any(studio in movie_data['Production Company'] 
                                              for studio in ['Warner Bros', 'Disney', 'Universal']) else 'no'
        
        # Filter selected features
        if self.selected_features:
            return {k: v for k, v in features.items() if k in self.selected_features}
        return features
    
    def fit(self, movie_data_list, labels):
        """Train the classifier"""
        self.class_counts = Counter()
        self.feature_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.feature_values = defaultdict(set)
        
        # Process training data
        for movie_data, label in zip(movie_data_list, labels):
            features = self.extract_movie_features(movie_data)
            self.class_counts[label] += 1
            for feature, value in features.items():
                self.feature_values[feature].add(value)
                self.feature_counts[label][feature][value] += 1
        
        self.features = list(self.feature_values.keys())
        self.total_samples = len(movie_data_list)
    
    def predict_proba(self, movie_data):
        """Predict class probabilities with logs and smoothing"""
        features = self.extract_movie_features(movie_data)
        best_class = None
        best_score = float('-inf')
        
        for cls in self.class_counts:
            # Prior probability with smoothing
            prior = (self.class_counts[cls] + 1) / (self.total_samples + len(self.class_counts))
            score = math.log(prior) if self.use_logs else prior
            
            # Feature likelihoods
            for feature in self.features:
                value = features.get(feature, None)
                if value is None:
                    continue
                    
                count = self.feature_counts[cls][feature].get(value, 0)
                num_values = len(self.feature_values[feature])
                
                # Apply Laplace smoothing
                likelihood = (count + 1) / (self.class_counts[cls] + num_values)
                score += math.log(likelihood) if self.use_logs else likelihood
            
            if score > best_score:
                best_score = score
                best_class = cls
                
        return best_class if best_class is not None else max(self.class_counts.keys(), key=lambda x: self.class_counts[x])

    def evaluate(self, test_movies, test_labels, verbose=False):
        """Evaluate classifier performance"""
        predictions = [self.predict_proba(movie) for movie in test_movies]
        accuracy = np.mean([p == t for p, t in zip(predictions, test_labels)])
        
        if verbose:
            print(f"Accuracy: {accuracy:.2f} ({len(test_movies)} movies)")
            print(classification_report(test_labels, predictions))
        return accuracy

def load_movie_data(filename):
    """Load movie data from CSV"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Dataset file {filename} not found")
    
    movies = []
    with open(filename, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                # Clean and convert data
                row['Gross Revenue (million)'] = float(row['Gross Revenue (million)']) if row['Gross Revenue (million)'] else 0.0
                movies.append(row)
            except (ValueError, KeyError) as e:
                print(f"Skipping row due to error: {e}")
    return movies

def calculate_success_labels(movies, success_threshold=None):
    """Calculate success labels based on revenue"""
    if 'Success' in movies[0]:
        return [m['Success'] for m in movies], None
    
    revenues = [m['Gross Revenue (million)'] for m in movies]
    threshold = success_threshold if success_threshold else np.median(revenues)
    return ['Success' if m['Gross Revenue (million)'] >= threshold else 'Failure' for m in movies], threshold

def random_split(movies, labels, train_ratio=0.7, random_state=42):
    """Create stratified train-test split"""
    random.seed(random_state)
    indices = list(range(len(movies)))
    random.shuffle(indices)
    
    split_point = int(train_ratio * len(movies))
    return (
        [movies[i] for i in indices[:split_point]],
        [labels[i] for i in indices[:split_point]],
        [movies[i] for i in indices[split_point:]],
        [labels[i] for i in indices[split_point:]]
    )

def feature_analysis(movies, labels, all_features):
    """Comprehensive feature evaluation with visualization"""
    results = {'single': [], 'pairs': [], 'triplets': []}
    
    # 1. Evaluate single features
    print("\n=== SINGLE FEATURE EVALUATION ===")
    for feature in all_features:
        acc = evaluate_feature_combination(movies, labels, [feature])
        results['single'].append((feature, acc))
        print(f"{feature:<20}: {acc:.3f}")
    
    # 2. Evaluate feature pairs (top 5 single features)
    top_features = [f[0] for f in sorted(results['single'], key=lambda x: x[1], reverse=True)[:5]]
    print("\n=== TOP FEATURE PAIRS ===")
    for f1, f2 in combinations(top_features, 2):
        acc = evaluate_feature_combination(movies, labels, [f1, f2])
        results['pairs'].append((f"{f1}+{f2}", acc))
        print(f"{f1}+{f2:<30}: {acc:.3f}")
    
    # 3. Visualize results
    plot_feature_results(results)
    return results

def evaluate_feature_combination(movies, labels, features, trials=10):
    """Evaluate a feature set with multiple random splits"""
    accuracies = []
    for _ in range(trials):
        train_m, train_l, test_m, test_l = random_split(movies, labels)
        clf = NaiveBayesClassifier(selected_features=features)
        clf.fit(train_m, train_l)
        acc = clf.evaluate(test_m, test_l)
        accuracies.append(acc)
    return np.mean(accuracies)

def plot_feature_results(results):
    """Visualize feature performance"""
    plt.figure(figsize=(12, 6))
    
    # Single features
    single_features = sorted(results['single'], key=lambda x: x[1])
    plt.barh([f[0] for f in single_features], 
             [f[1] for f in single_features],
             color='skyblue', label='Single Features')
    
    # Annotate best single feature
    best_single = max(results['single'], key=lambda x: x[1])
    plt.annotate(f'Best: {best_single[1]:.2f}', 
                 xy=(best_single[1], best_single[0]),
                 xytext=(5, 0), textcoords='offset points')
    
    plt.title('Feature Performance Analysis')
    plt.xlabel('Accuracy')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_performance.png', dpi=300)
    plt.show()

def learning_curve_analysis(movies, labels, best_features):
    """Generate learning curve with confidence intervals"""
    train_sizes = np.linspace(0.1, 0.9, 9)  # 10% to 90% of data
    mean_acc = []
    std_acc = []
    
    for size in train_sizes:
        accs = []
        for _ in range(10):  # 10 trials per size
            train_m, train_l, test_m, test_l = random_split(movies, labels, train_ratio=size)
            clf = NaiveBayesClassifier(selected_features=best_features)
            clf.fit(train_m, train_l)
            acc = clf.evaluate(test_m, test_l)
            accs.append(acc)
        
        mean_acc.append(np.mean(accs))
        std_acc.append(np.std(accs))
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, mean_acc, 'o-', label='Test Accuracy')
    plt.fill_between(train_sizes, 
                    np.array(mean_acc) - np.array(std_acc),
                    np.array(mean_acc) + np.array(std_acc),
                    alpha=0.2)
    
    # Mark optimal training size
    opt_size = train_sizes[np.argmax(mean_acc)]
    plt.axvline(x=opt_size, color='r', linestyle='--', 
                label=f'Optimal Size: {opt_size:.0%}')
    
    plt.title('Learning Curve with Confidence Intervals')
    plt.xlabel('Training Set Proportion')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig('learning_curve.png', dpi=300)
    plt.show()
    
    return opt_size

def main():
    print("=== ENHANCED MOVIE SUCCESS PREDICTOR ===")
    
    # Load and prepare data
    try:
        movies = load_movie_data('15movies_with_success.csv')
        labels, _ = calculate_success_labels(movies)
        print(f"Loaded {len(movies)} movies with {sum(1 for l in labels if l == 'Success')} successes")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Get all available features
    sample_features = NaiveBayesClassifier().extract_movie_features(movies[0])
    all_features = list(sample_features.keys())
    print(f"\nAvailable features ({len(all_features)}): {all_features}")
    
    # Comprehensive feature analysis
    results = feature_analysis(movies, labels, all_features)
    best_feature = max(results['single'], key=lambda x: x[1])[0]
    
    # Learning curve analysis
    optimal_size = learning_curve_analysis(movies, labels, [best_feature])
    
    # Final evaluation with best features
    print("\n=== FINAL EVALUATION ===")
    train_m, train_l, test_m, test_l = random_split(movies, labels)
    clf = NaiveBayesClassifier(selected_features=[best_feature])
    clf.fit(train_m, train_l)
    accuracy = clf.evaluate(test_m, test_l, verbose=True)
    
    print("\n=== RESULTS SUMMARY ===")
    print(f"Best single feature: {best_feature}")
    print(f"Maximum accuracy: {accuracy:.2f}")
    print(f"Optimal training size: {optimal_size:.0%} of dataset")
    print("Visualizations saved to:")
    print("- feature_performance.png")
    print("- learning_curve.png")

if __name__ == "__main__":
    main()