import csv
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from collections import defaultdict

class SVMMovieClassifier:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.svm = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42)
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.features = []
        
    def extract_movie_features(self, movie_data):
        """Extract features from movie data (same as Naive Bayes for fair comparison)"""
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
    
    def encode_features(self, features_list, fit_encoders=False):
        """Convert categorical features to numerical values"""
        if not features_list:
            return np.array([])
            
        # Get feature names from first sample
        if not self.features:
            self.features = list(features_list[0].keys())
        
        encoded_data = []
        
        for features in features_list:
            encoded_row = []
            for feature in self.features:
                value = features[feature]
                
                if fit_encoders:
                    if feature not in self.label_encoders:
                        self.label_encoders[feature] = LabelEncoder()
                        # Fit on all possible values for this feature from training data
                        all_values = [f[feature] for f in features_list]
                        self.label_encoders[feature].fit(all_values)
                
                # Handle unseen values during prediction
                try:
                    encoded_value = self.label_encoders[feature].transform([value])[0]
                except (KeyError, ValueError):
                    # If unseen value, use the most frequent class (index 0)
                    encoded_value = 0
                
                encoded_row.append(encoded_value)
            encoded_data.append(encoded_row)
        
        return np.array(encoded_data)
    
    def fit(self, movie_data_list, labels):
        """Train the SVM classifier"""
        # Extract features
        features_list = [self.extract_movie_features(movie) for movie in movie_data_list]
        
        # Encode categorical features to numerical
        X = self.encode_features(features_list, fit_encoders=True)
        
        # Scale features for SVM
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert labels to binary (1 for Success, 0 for Failure)
        y = [1 if label == "Success" else 0 for label in labels]
        
        # Train SVM
        self.svm.fit(X_scaled, y)
        
        return self
    
    def predict_proba(self, movie_data, verbose=False):
        """Predict class probabilities for a movie"""
        features = self.extract_movie_features(movie_data)
        
        if verbose:
            print(f"\nSVM Prediction for movie: '{movie_data['Title']}'")
            print(f"Extracted features: {features}")
        
        # Encode features
        X = self.encode_features([features], fit_encoders=False)
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities
        probabilities = self.svm.predict_proba(X_scaled)[0]
        failure_prob, success_prob = probabilities
        
        prediction = "Success" if success_prob > failure_prob else "Failure"
        
        if verbose:
            print(f"SVM Probabilities:")
            print(f"  P(Failure) = {failure_prob:.6f}")
            print(f"  P(Success) = {success_prob:.6f}")
            print(f"Prediction: {prediction}")
        
        return prediction, {"Failure": failure_prob, "Success": success_prob}
    
    def predict(self, movie_data):
        """Predict class for a movie"""
        prediction, _ = self.predict_proba(movie_data)
        return prediction
    
    def evaluate(self, test_movies, test_labels, verbose=False):
        """Evaluate SVM classifier with comprehensive metrics"""
        predictions = []

        if verbose:
            print("=== SVM EVALUATION RESULTS ===")

        for i, (movie_data, true_label) in enumerate(zip(test_movies, test_labels)):
            predicted_label = self.predict(movie_data)
            predictions.append(predicted_label)

            if verbose:
                status = "✓" if predicted_label == true_label else "✗"
                print(f"{status} '{movie_data['Title']}' -> Predicted: {predicted_label}, Actual: {true_label}")

        # Calculate all metrics
        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions, pos_label="Success", average='binary')
        recall = recall_score(test_labels, predictions, pos_label="Success", average='binary')
        f1 = f1_score(test_labels, predictions, pos_label="Success", average='binary')
        
        # Display metrics
        print(f"\n=== PERFORMANCE METRICS ===")
        print(f"Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"Precision: {precision:.3f} ({precision*100:.1f}%)")
        print(f"Recall:    {recall:.3f} ({recall*100:.1f}%)")
        print(f"F1-Score:  {f1:.3f} ({f1*100:.1f}%)")

        # Confusion matrix
        print("\n=== CONFUSION MATRIX ===")
        cm = confusion_matrix(test_labels, predictions, labels=["Success", "Failure"])
        cm_df = pd.DataFrame(cm, 
                           index=["Actual Success", "Actual Failure"], 
                           columns=["Predicted Success", "Predicted Failure"])
        print(cm_df)
        
        # Additional confusion matrix breakdown
        tn, fp, fn, tp = cm.ravel()
        print(f"\nConfusion Matrix Breakdown:")
        print(f"True Positives (TP):  {tp}")
        print(f"True Negatives (TN):  {tn}")
        print(f"False Positives (FP): {fp}")
        print(f"False Negatives (FN): {fn}")

        # Classification report
        print("\n=== DETAILED CLASSIFICATION REPORT ===")
        print(classification_report(test_labels, predictions, digits=3))

        # Return metrics dictionary
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'predictions': predictions
        }
        
        return metrics


def load_movie_data_with_success(filename):
    """Load movie data from CSV file with success labels"""
    movies = []
    labels = []
    
    with open(filename, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                # Convert numerical fields
                row['Year'] = int(row['Year'])
                row['Runtime (min)'] = int(row['Runtime (min)'])
                row['Gross Revenue (million)'] = float(row['Gross Revenue (million)']) if row['Gross Revenue (million)'] else 0.0
                
                # Extract success label
                success_label = "Success" if int(row['Success']) == 1 else "Failure"
                labels.append(success_label)
                
                movies.append(row)
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not process row {row}. Error: {e}")
                continue
    
    return movies, labels


def hyperparameter_tuning(X_train, y_train, verbose=False):
    """Perform hyperparameter tuning for SVM"""
    print("\nPerforming hyperparameter tuning...")
    
    # Define parameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear', 'poly']
    }
    
    # Create SVM classifier
    svm = SVC(probability=True, random_state=42)
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', verbose=1 if verbose else 0)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
    
    return grid_search.best_params_


def main():
    print("=== SVM MOVIE SUCCESS CLASSIFIER ===")
    
    # Load data
    filename = '30movies_with_success.csv'
    
    try:
        movie_dataset, labels = load_movie_data_with_success(filename)
        print(f"Loaded {len(movie_dataset)} movies from {filename}")
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found. Please ensure it's in the same directory.")
        return
    
    # Display dataset statistics
    success_count = labels.count("Success")
    failure_count = labels.count("Failure")
    print(f"Success: {success_count} movies ({success_count/len(labels)*100:.1f}%)")
    print(f"Failure: {failure_count} movies ({failure_count/len(labels)*100:.1f}%)")
    
    # Split data (75% train, 25% test)
    split_point = int(0.75 * len(movie_dataset))
    train_movies = movie_dataset[:split_point]
    train_labels = labels[:split_point]
    test_movies = movie_dataset[split_point:]
    test_labels = labels[split_point:]
    
    print(f"\nTraining set: {len(train_movies)} movies")
    print(f"Test set: {len(test_movies)} movies")
    
    # Test different SVM configurations
    configs = [
        ("SVM - RBF Kernel", 'rbf', 1.0, 'scale'),
        ("SVM - Linear Kernel", 'linear', 1.0, 'scale'),
        ("SVM - Polynomial Kernel", 'poly', 1.0, 'scale'),
        ("SVM - RBF High C", 'rbf', 10.0, 'scale'),
    ]
    
    best_accuracy = 0
    best_config = None
    best_classifier = None
    best_metrics = None
    
    for config_name, kernel, C, gamma in configs:
        print(f"\n{'='*60}")
        print(f"CONFIGURATION: {config_name}")
        print('='*60)
        
        # Create and train classifier
        classifier = SVMMovieClassifier(kernel=kernel, C=C, gamma=gamma)
        classifier.fit(train_movies, train_labels)
        
        # Evaluate
        metrics = classifier.evaluate(test_movies, test_labels, verbose=True)
        
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            best_config = config_name
            best_classifier = classifier
            best_metrics = metrics
        
        # Show detailed prediction for one example
        if test_movies:
            example_movie = test_movies[0]
            print(f"\nDetailed prediction example:")
            classifier.predict_proba(example_movie, verbose=True)
    
    # Hyperparameter tuning for best performance
    print(f"\n{'='*60}")
    print("HYPERPARAMETER TUNING")
    print('='*60)
    
    # Extract and encode features for hyperparameter tuning
    train_features = [best_classifier.extract_movie_features(movie) for movie in train_movies]
    X_train = best_classifier.encode_features(train_features, fit_encoders=True)
    X_train_scaled = best_classifier.scaler.fit_transform(X_train)
    y_train = [1 if label == "Success" else 0 for label in train_labels]
    
    best_params = hyperparameter_tuning(X_train_scaled, y_train)
    
    # Train final classifier with best parameters
    final_classifier = SVMMovieClassifier(
        kernel=best_params['kernel'],
        C=best_params['C'],
        gamma=best_params['gamma']
    )
    final_classifier.fit(train_movies, train_labels)
    
    print(f"\n{'='*60}")
    print("FINAL OPTIMIZED SVM RESULTS")
    print('='*60)
    
    final_metrics = final_classifier.evaluate(test_movies, test_labels, verbose=True)
    
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print('='*60)
    print(f"Best Basic Configuration: {best_config}")
    print(f"Best Basic Metrics:")
    print(f"  Accuracy:  {best_metrics['accuracy']:.3f} ({best_metrics['accuracy']*100:.1f}%)")
    print(f"  Precision: {best_metrics['precision']:.3f} ({best_metrics['precision']*100:.1f}%)")
    print(f"  Recall:    {best_metrics['recall']:.3f} ({best_metrics['recall']*100:.1f}%)")
    print(f"  F1-Score:  {best_metrics['f1_score']:.3f} ({best_metrics['f1_score']*100:.1f}%)")
    
    print(f"\nOptimized SVM Metrics:")
    print(f"  Accuracy:  {final_metrics['accuracy']:.3f} ({final_metrics['accuracy']*100:.1f}%)")
    print(f"  Precision: {final_metrics['precision']:.3f} ({final_metrics['precision']*100:.1f}%)")
    print(f"  Recall:    {final_metrics['recall']:.3f} ({final_metrics['recall']*100:.1f}%)")
    print(f"  F1-Score:  {final_metrics['f1_score']:.3f} ({final_metrics['f1_score']*100:.1f}%)")
    
    # Performance comparison table
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON TABLE")
    print('='*60)
    comparison_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Best Basic SVM': [
            f"{best_metrics['accuracy']:.3f}",
            f"{best_metrics['precision']:.3f}",
            f"{best_metrics['recall']:.3f}",
            f"{best_metrics['f1_score']:.3f}"
        ],
        'Optimized SVM': [
            f"{final_metrics['accuracy']:.3f}",
            f"{final_metrics['precision']:.3f}",
            f"{final_metrics['recall']:.3f}",
            f"{final_metrics['f1_score']:.3f}"
        ]
    })
    print(comparison_df.to_string(index=False))
    
    # Feature importance analysis (for linear kernel)
    if best_params['kernel'] == 'linear':
        print(f"\nFeature Importance (Linear SVM):")
        print("-" * 40)
        coefficients = final_classifier.svm.coef_[0]
        for i, feature in enumerate(final_classifier.features):
            importance = abs(coefficients[i])
            print(f"{feature}: {importance:.4f}")
    
    # Test on new movies
    new_movies = [
        {"Title": "Avatar 3", "Year": 2011, "Genre": "Sci-Fi", "Director": "James Cameron", "Runtime (min)": 150, "Success": 0},
        {"Title": "The Comedy Show", "Year": 2011, "Genre": "Comedy", "Director": "Unknown Director", "Runtime (min)": 95, "Success": 0},
        {"Title": "Dark Knight Returns", "Year": 2011, "Genre": "Action", "Director": "Christopher Nolan", "Runtime (min)": 145, "Success": 0},
    ]
    
    print(f"\nPredictions for hypothetical new movies:")
    print("-" * 50)
    
    for movie in new_movies:
        # Add missing fields
        movie.update({
            "Production Company": "Unknown",
            "Country of Origin": "USA",
            "Original Language": "English",
            "Gross Revenue (million)": 0
        })
        
        prediction = final_classifier.predict(movie)
        print(f"'{movie['Title']}' ({movie['Genre']}, {movie['Director']}) -> {prediction}")


if __name__ == "__main__":
    main()