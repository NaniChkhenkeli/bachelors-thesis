import csv
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np

class MovieClassifier:
    def __init__(self):
        self.model = DecisionTreeClassifier(random_state=42)
        self.preprocessor = None
        self.label_encoder = LabelEncoder()
        self.features = []
        
    def extract_movie_features(self, movie_data):
        """Enhanced feature extraction with more sophisticated features"""
        features = {}
        
        title = movie_data['Title']
        year = int(movie_data['Year'])
        genre = movie_data['Genre']
        director = movie_data['Director']
        runtime = int(movie_data['Runtime (min)'])
        
        # Enhanced title analysis
        title_lower = title.lower()
        features['title_length'] = len(title)
        features['word_count'] = len(title.split())
        features['has_number'] = int(any(char.isdigit() for char in title))
        features['has_colon'] = int(':' in title)
        features['starts_with_the'] = int(title_lower.startswith('the '))
        
        # Genre features - consider multiple genres
        genres = [g.strip() for g in genre.split(',')]
        features['genre'] = genres[0]  # Use first genre as main category
        features['genre_count'] = len(genres)
        features['is_action'] = int('Action' in genres)
        features['is_comedy'] = int('Comedy' in genres)
        features['is_drama'] = int('Drama' in genres)
        
        # Temporal features
        features['year'] = year
        features['runtime'] = runtime
        
        # Director features
        experienced_directors = ['Steven Spielberg', 'Christopher Nolan', 
                               'Michael Bay', 'David Fincher', 'James Cameron']
        features['experienced_director'] = int(director in experienced_directors)
        
        # Franchise/sequel indicators
        sequel_words = ['2', 'ii', 'iii', 'iv', 'part', 'rise', 'dark', 
                       'breaking dawn', 'returns', 'revenge']
        features['likely_sequel'] = int(any(word in title_lower for word in sequel_words))
        
        return features

    def fit(self, movie_data_list, labels):
        """Enhanced training with better feature processing"""
        # Extract features
        features_list = [self.extract_movie_features(movie) for movie in movie_data_list]
        self.features = list(features_list[0].keys()) if features_list else []
        
        # Convert to DataFrame
        df = pd.DataFrame(features_list)
        
        # Encode labels
        y = self.label_encoder.fit_transform(labels)
        
        # Identify feature types
        categorical_features = ['genre']
        numeric_features = ['title_length', 'word_count', 'runtime', 'year', 'genre_count']
        binary_features = [f for f in df.columns if f.startswith(('has_', 'is_', 'likely_', 'experienced_'))]
        
        # Create preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numeric_features + binary_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        
        # Preprocess features
        X = self.preprocessor.fit_transform(df)
        
        # Hyperparameter tuning with GridSearch
        param_grid = {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
        
        grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42),
                                 param_grid, cv=5, scoring='f1')
        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_
        
        print(f"\nBest Decision Tree parameters: {grid_search.best_params_}")
        
        # Advanced tree visualization
        self.plot_decision_tree(X, y)

    def plot_decision_tree(self, X, y):
        """Create advanced visualization of the decision tree"""
        plt.figure(figsize=(24, 12))
        
        plot_tree(self.model,
                 feature_names=self.preprocessor.get_feature_names_out(),
                 class_names=self.label_encoder.classes_,
                 filled=True,
                 rounded=True,
                 proportion=True,
                 impurity=False,
                 node_ids=True,
                 fontsize=10,
                 max_depth=3)  # Limit depth for better visualization
        
        plt.title("Advanced Decision Tree Visualization for Movie Success Prediction\n"
                 f"Depth: {self.model.get_depth()}, "
                 f"Leaves: {self.model.get_n_leaves()}\n"
                 f"Best Parameters: {self.model.get_params()}", 
                 fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig('30movie_success_decision_tree.png', dpi=300)
        plt.show()

    def predict(self, movie_data):
        """Predict class for a movie"""
        features = self.extract_movie_features(movie_data)
        df = pd.DataFrame([features])
        X = self.preprocessor.transform(df)
        prediction = self.model.predict(X)
        return self.label_encoder.inverse_transform(prediction)[0]

    def evaluate(self, test_movies, test_labels, verbose=False):
        """Enhanced evaluation with more metrics"""
        predictions = [self.predict(movie) for movie in test_movies]
        
        # Calculate metrics
        y_true = self.label_encoder.transform(test_labels)
        y_pred = self.label_encoder.transform(predictions)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        cr = classification_report(y_true, y_pred, 
                                 target_names=self.label_encoder.classes_)
        
        if verbose:
            print("\n=== MODEL EVALUATION RESULTS ===")
            print(f"\nConfusion Matrix:\n{cm}")
            print(f"\nClassification Report:\n{cr}")
            print(f"\nAccuracy: {accuracy:.3f}")
            print(f"Precision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")
            print(f"F1 Score: {f1:.3f}")
            
            print("\nDetailed Predictions:")
            for i, (movie_data, true_label) in enumerate(zip(test_movies, test_labels)):
                status = "✓" if predictions[i] == true_label else "✗"
                print(f"{status} '{movie_data['Title']}' (${movie_data['Gross Revenue (million)']}M) -> Predicted: {predictions[i]}, Actual: {true_label}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': predictions
        }

def load_movie_data(filename):
    """Load movie data from CSV file with error handling"""
    movies = []
    with open(filename, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                # Convert numerical fields with error handling
                row['Year'] = int(row['Year'])
                row['Runtime (min)'] = int(row['Runtime (min)'])
                
                # Handle missing revenue values
                revenue = row.get('Gross Revenue (million)', '0').strip()
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
    
    print(f"\n{'='*50}")
    print(f"Average Revenue: ${average_revenue:.1f} million")
    print(f"{'='*50}\n")
    
    labels = []
    success_count = 0
    
    print("{:<40} {:>10} {:>15}".format("Movie Title", "Revenue", "Status"))
    print("-"*70)
    for movie in movies:
        revenue = movie["Gross Revenue (million)"]
        if revenue > average_revenue:
            labels.append("Success")
            success_count += 1
            status = "SUCCESS"
        else:
            labels.append("Failure")
            status = "FAILURE"
        
        print(f"{movie['Title'][:37]:<40} ${revenue:>6.1f}M {'':>5} {status}")
    
    print(f"\nSuccessful movies: {success_count}/{len(movies)} ({success_count/len(movies)*100:.1f}%)")
    print(f"Failed movies: {len(movies)-success_count}/{len(movies)} ({(len(movies)-success_count)/len(movies)*100:.1f}%)")
    
    return labels, average_revenue

def analyze_title_lengths(movies, labels):
    """Calculate average title length for successful vs. unsuccessful movies"""
    success_lengths = []
    failure_lengths = []
    
    for movie, label in zip(movies, labels):
        title_len = len(movie['Title'])
        if label == "Success":
            success_lengths.append(title_len)
        else:
            failure_lengths.append(title_len)
    
    avg_success_len = np.mean(success_lengths) if success_lengths else 0
    avg_failure_len = np.mean(failure_lengths) if failure_lengths else 0
    
    print("\n=== Title Length Analysis ===")
    print(f"Average title length (Success): {avg_success_len:.1f} chars")
    print(f"Average title length (Failure): {avg_failure_len:.1f} chars")
    print(f"Overall average title length: {np.mean(success_lengths + failure_lengths):.1f} chars")
    
    # Plot distribution
    plt.figure(figsize=(10, 5))
    plt.hist(success_lengths, bins=20, alpha=0.5, label="Success", color='green')
    plt.hist(failure_lengths, bins=20, alpha=0.5, label="Failure", color='red')
    plt.axvline(15, color='black', linestyle='--', label="Tree Split (≤28)")
    plt.xlabel("Title Length (chars)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Title Lengths for Successful vs. Failed Movies")
    plt.legend()
    plt.show()
    
    return avg_success_len, avg_failure_len

def main():
    print("=== MOVIE SUCCESS PREDICTION SYSTEM (DECISION TREE) ===")
    
    # Load data
    input_filename = '270movies_with_success.csv'  # Replace with your file
    movie_dataset = load_movie_data(input_filename)
    labels, avg_revenue = calculate_success_labels(movie_dataset)
    
    # Analyze title lengths
    avg_success_len, avg_failure_len = analyze_title_lengths(movie_dataset, labels)
    
    # Stratified train-test split
    X = pd.DataFrame(movie_dataset)
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Convert back to list of dicts
    train_movies = X_train.to_dict('records')
    test_movies = X_test.to_dict('records')
    
    # Train and evaluate classifier
    classifier = MovieClassifier()
    print("\nTraining Decision Tree model...")
    classifier.fit(train_movies, y_train)
    
    print("\nEvaluating model on test set...")
    eval_results = classifier.evaluate(test_movies, y_test, verbose=True)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL MODEL PERFORMANCE SUMMARY")
    print('='*60)
    print(f"Accuracy: {eval_results['accuracy']:.1%}")
    print(f"Precision: {eval_results['precision']:.3f}")
    print(f"Recall: {eval_results['recall']:.3f}")
    print(f"F1 Score: {eval_results['f1']:.3f}")
    print("\nDecision tree visualization saved as 'movie_success_decision_tree.png'")

if __name__ == "__main__":
    main()