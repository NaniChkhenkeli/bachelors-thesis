import csv
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import random

class NeuralNetworkClassifier:
    def __init__(self, hidden_sizes=[64, 32], learning_rate=0.001, epochs=100, 
                 batch_size=32, dropout_rate=0.2, use_feature_selection=True, 
                 selected_features=None, random_seed=42, use_regression_output=True):
        """
        Neural Network Classifier with optional regression output for success scoring
        
        Args:
            hidden_sizes: List of hidden layer sizes
            learning_rate: Learning rate for gradient descent
            epochs: Number of training epochs
            batch_size: Batch size for training
            dropout_rate: Dropout rate for regularization
            use_feature_selection: Whether to use feature selection
            selected_features: Specific features to use
            random_seed: Random seed for reproducibility
            use_regression_output: Whether to add regression output for success scoring
        """
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.use_feature_selection = use_feature_selection
        self.selected_features = selected_features
        self.random_seed = random_seed
        self.use_regression_output = use_regression_output
        
        # Set random seeds for reproducibility
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Initialize components
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.weights = []
        self.biases = []
        self.training_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        self.revenue_mean = 0
        self.revenue_std = 1
        
        # Best features identified from analysis
        self.best_features = [
            'genre', 'runtime_category', 'director', 'production_company',
            'major_studio', 'year_category', 'english_language', 'action_genre'
        ]
    
    def extract_movie_features(self, movie_data):
        """Extract features from movie data"""
        features = {}
        
        # Extract basic fields
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
        
        # Direct features
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
        
        # Numerical features
        features['runtime_numeric'] = runtime
        features['year_numeric'] = year
        
        # Director-based features
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
        
        # Use selected features if specified
        if self.use_feature_selection:
            selected = self.selected_features if self.selected_features else self.best_features
            features = {k: v for k, v in features.items() if k in selected}
        
        return features
    
    def prepare_features(self, movie_data_list, is_training=True):
        """Convert movie data to numerical feature matrix"""
        feature_list = []
        
        # Extract features for all movies
        for movie_data in movie_data_list:
            features = self.extract_movie_features(movie_data)
            feature_list.append(features)
        
        if not feature_list:
            return np.array([])
        
        # Get feature names from first sample
        if is_training:
            self.feature_names = list(feature_list[0].keys())
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(feature_list)
        
        # Handle categorical and numerical features
        X = np.zeros((len(df), 0))
        
        for feature in self.feature_names:
            if feature in df.columns:
                values = df[feature].fillna('Unknown')
                
                # Check if feature is numerical
                if feature.endswith('_numeric'):
                    # Numerical feature - use as is
                    numerical_values = pd.to_numeric(values, errors='coerce').fillna(0)
                    X = np.column_stack([X, numerical_values.values.reshape(-1, 1)]) if X.size > 0 else numerical_values.values.reshape(-1, 1)
                else:
                    # Categorical feature - encode
                    if is_training:
                        if feature not in self.label_encoders:
                            self.label_encoders[feature] = LabelEncoder()
                            encoded = self.label_encoders[feature].fit_transform(values.astype(str))
                        else:
                            encoded = self.label_encoders[feature].transform(values.astype(str))
                    else:
                        if feature in self.label_encoders:
                            # Handle unseen categories
                            encoded = []
                            for val in values.astype(str):
                                if val in self.label_encoders[feature].classes_:
                                    encoded.append(self.label_encoders[feature].transform([val])[0])
                                else:
                                    encoded.append(0)  # Default to first class for unseen categories
                            encoded = np.array(encoded)
                        else:
                            encoded = np.zeros(len(values))
                    
                    X = np.column_stack([X, encoded.reshape(-1, 1)]) if X.size > 0 else encoded.reshape(-1, 1)
            else:
                # Missing feature - add zeros
                X = np.column_stack([X, np.zeros(len(df)).reshape(-1, 1)]) if X.size > 0 else np.zeros(len(df)).reshape(-1, 1)
        
        # Scale features
        if is_training:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        return X
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)
    
    def initialize_weights(self, input_size):
        """Initialize network weights and biases for both classification and regression"""
        self.weights = []
        self.biases = []
        
        # Shared hidden layers
        layer_sizes = [input_size] + self.hidden_sizes
        
        for i in range(len(layer_sizes) - 1):
            # Xavier/Glorot initialization
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            bias = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)
        
        # Classification output (binary)
        self.weights.append(np.random.randn(layer_sizes[-1], 1) * np.sqrt(2.0 / layer_sizes[-1]))
        self.biases.append(np.zeros((1, 1)))
        
        # Regression output (optional)
        if self.use_regression_output:
            self.weights.append(np.random.randn(layer_sizes[-1], 1) * np.sqrt(2.0 / layer_sizes[-1]))
            self.biases.append(np.zeros((1, 1)))
    
    def forward_pass(self, X, training=True):
        """Forward propagation through the network"""
        activations = [X]
        z_values = []
        
        # Shared hidden layers
        for i, (W, b) in enumerate(zip(self.weights[:-2], self.biases[:-2])):
            z = np.dot(activations[-1], W) + b
            z_values.append(z)
            a = self.relu(z)
            # Apply dropout during training
            if training and self.dropout_rate > 0:
                dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, a.shape) / (1 - self.dropout_rate)
                a *= dropout_mask
            activations.append(a)
        
        # Classification output
        z_class = np.dot(activations[-1], self.weights[-2]) + self.biases[-2]
        a_class = self.sigmoid(z_class)
        activations.append(a_class)
        z_values.append(z_class)
        
        # Regression output (if enabled)
        if self.use_regression_output:
            z_reg = np.dot(activations[-2], self.weights[-1]) + self.biases[-1]
            a_reg = z_reg  # Linear activation for regression
            activations.append(a_reg)
            z_values.append(z_reg)
        
        return activations, z_values
    
    def compute_loss(self, y_true_class, y_pred_class, y_true_reg=None, y_pred_reg=None):
        """Compute combined classification and regression loss"""
        # Classification loss (binary cross-entropy)
        y_pred_class = np.clip(y_pred_class, 1e-15, 1 - 1e-15)
        class_loss = -np.mean(y_true_class * np.log(y_pred_class) + (1 - y_true_class) * np.log(1 - y_pred_class))
        
        # Regression loss (MSE, if regression output is enabled)
        reg_loss = 0.0
        if self.use_regression_output and y_true_reg is not None:
            reg_loss = np.mean((y_true_reg - y_pred_reg) ** 2)
        
        return class_loss + 0.1 * reg_loss  # Weighted combination
    
    def backward_pass(self, X, y_class, y_reg, activations, z_values):
        """Backpropagation to compute gradients"""
        m = X.shape[0]
        gradients_w = [np.zeros_like(W) for W in self.weights]
        gradients_b = [np.zeros_like(b) for b in self.biases]
        
        # Classification output error
        delta_class = activations[-2] - y_class.reshape(-1, 1)
        gradients_w[-2] = np.dot(activations[-3].T, delta_class) / m
        gradients_b[-2] = np.sum(delta_class, axis=0, keepdims=True) / m
        
        # Regression output error (if enabled)
        if self.use_regression_output:
            delta_reg = (activations[-1] - y_reg.reshape(-1, 1)) / m
            gradients_w[-1] = np.dot(activations[-3].T, delta_reg)
            gradients_b[-1] = np.sum(delta_reg, axis=0, keepdims=True)
        
        # Backpropagate through shared layers
        delta = (np.dot(delta_class, self.weights[-2].T) + 
                (0.1 * np.dot(delta_reg, self.weights[-1].T) if self.use_regression_output else 0)) * self.relu_derivative(activations[-3])
        
        for i in range(len(self.weights) - 3, -1, -1):
            gradients_w[i] = np.dot(activations[i].T, delta) / m
            gradients_b[i] = np.sum(delta, axis=0, keepdims=True) / m
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(activations[i])
        
        return gradients_w, gradients_b
    
    def update_weights(self, gradients_w, gradients_b):
        """Update weights and biases using gradients"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]
    
    def fit(self, movie_data_list, labels, revenues=None, validation_data=None):
        """Train the neural network"""
        print(f"Training Neural Network with {len(movie_data_list)} samples...")
        
        # Prepare features
        X = self.prepare_features(movie_data_list, is_training=True)
        y_class = np.array([1 if label == "Success" else 0 for label in labels])
        
        # Prepare regression targets (if enabled)
        y_reg = None
        if self.use_regression_output and revenues is not None:
            y_reg = np.array(revenues)
            self.revenue_mean, self.revenue_std = y_reg.mean(), y_reg.std()
            y_reg = (y_reg - self.revenue_mean) / self.revenue_std  # Normalize
        
        # Initialize weights
        self.initialize_weights(X.shape[1])
        
        # Prepare validation data if provided
        X_val, y_val_class, y_val_reg = None, None, None
        if validation_data:
            val_movies, val_labels, val_revenues = validation_data
            X_val = self.prepare_features(val_movies, is_training=False)
            y_val_class = np.array([1 if label == "Success" else 0 for label in val_labels])
            if self.use_regression_output and val_revenues is not None:
                y_val_reg = (np.array(val_revenues) - self.revenue_mean) / self.revenue_std
        
        print(f"Input features: {X.shape[1]}")
        print(f"Network architecture: {X.shape[1]} -> {' -> '.join(map(str, self.hidden_sizes))} -> 1 (class) {'-> 1 (reg)' if self.use_regression_output else ''}")
        
        # Training loop
        for epoch in range(self.epochs):
            # Shuffle data
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_class_shuffled = y_class[indices]
            y_reg_shuffled = y_reg[indices] if y_reg is not None else None
            
            # Mini-batch training
            epoch_loss = 0
            num_batches = 0
            
            for i in range(0, len(X), self.batch_size):
                batch_X = X_shuffled[i:i + self.batch_size]
                batch_y_class = y_class_shuffled[i:i + self.batch_size]
                batch_y_reg = y_reg_shuffled[i:i + self.batch_size] if y_reg_shuffled is not None else None
                
                # Forward pass
                activations, z_values = self.forward_pass(batch_X, training=True)
                
                # Compute loss
                batch_loss = self.compute_loss(
                    batch_y_class, activations[-2].flatten(),
                    batch_y_reg, activations[-1].flatten() if self.use_regression_output else None
                )
                epoch_loss += batch_loss
                num_batches += 1
                
                # Backward pass
                gradients_w, gradients_b = self.backward_pass(
                    batch_X, batch_y_class, batch_y_reg, activations, z_values
                )
                
                # Update weights
                self.update_weights(gradients_w, gradients_b)
            
            # Calculate epoch metrics
            avg_loss = epoch_loss / num_batches
            
            # Training accuracy
            train_predictions = self.predict_proba(movie_data_list)
            train_accuracy = np.mean((train_predictions > 0.5) == (y_class == 1))
            
            self.training_history['loss'].append(avg_loss)
            self.training_history['accuracy'].append(train_accuracy)
            
            # Validation metrics
            val_accuracy = None
            val_loss = None
            if validation_data:
                val_predictions = []
                val_scores = []
                for movie in val_movies:
                    val_predictions.append(self.predict_proba_single(movie))
                    if self.use_regression_output:
                        val_scores.append(self.predict_success_score_single(movie))
                val_predictions = np.array(val_predictions)
                val_accuracy = np.mean((val_predictions > 0.5) == (y_val_class == 1))
                
                if self.use_regression_output and y_val_reg is not None:
                    val_scores = np.array(val_scores)
                    val_reg_loss = np.mean((val_scores - y_val_reg) ** 2)
                    val_loss = avg_loss + 0.1 * val_reg_loss
                    self.training_history['val_loss'].append(val_loss)
                
                self.training_history['val_accuracy'].append(val_accuracy)
            
            # Print progress
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                val_str = f", Val Acc: {val_accuracy:.3f}" if val_accuracy is not None else ""
                val_loss_str = f", Val Loss: {val_loss:.4f}" if val_loss is not None else ""
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}{val_loss_str}, Train Acc: {train_accuracy:.3f}{val_str}")
        
        print("Training completed!")
    
    def predict_proba_single(self, movie_data):
        """Predict success probability for a single movie"""
        X = self.prepare_features([movie_data], is_training=False)
        if X.size == 0:
            return 0.5  # Default probability
        
        activations, _ = self.forward_pass(X, training=False)
        return activations[-2][0, 0]  # Classification output
    
    def predict_proba(self, movie_data_list):
        """Predict probabilities for a list of movies"""
        if not movie_data_list:
            return np.array([])
        
        X = self.prepare_features(movie_data_list, is_training=False)
        if X.size == 0:
            return np.full(len(movie_data_list), 0.5)
        
        activations, _ = self.forward_pass(X, training=False)
        return activations[-2].flatten()  # Classification output
    
    def predict_success_score_single(self, movie_data):
        """Predict success score for a single movie"""
        if not self.use_regression_output:
            raise ValueError("Regression output not enabled during initialization")
        
        X = self.prepare_features([movie_data], is_training=False)
        if X.size == 0:
            return self.revenue_mean  # Default to mean
        
        activations, _ = self.forward_pass(X, training=False)
        return activations[-1][0, 0] * self.revenue_std + self.revenue_mean  # Denormalize
    
    def predict_success_score(self, movie_data_list):
        """Predict success scores for a list of movies"""
        if not self.use_regression_output:
            raise ValueError("Regression output not enabled during initialization")
        
        if not movie_data_list:
            return np.array([])
        
        X = self.prepare_features(movie_data_list, is_training=False)
        if X.size == 0:
            return np.full(len(movie_data_list), self.revenue_mean)
        
        activations, _ = self.forward_pass(X, training=False)
        return activations[-1].flatten() * self.revenue_std + self.revenue_mean  # Denormalize
    
    def predict(self, movie_data_list):
        """Predict classes for a list of movies"""
        probabilities = self.predict_proba(movie_data_list)
        return ["Success" if prob > 0.5 else "Failure" for prob in probabilities]
    
    def evaluate(self, test_movies, test_labels, test_revenues=None, verbose=False):
        """Evaluate the classifier"""
        if not test_movies:
            return 0.0, [], []
        
        predictions = self.predict(test_movies)
        probabilities = self.predict_proba(test_movies)
        success_scores = self.predict_success_score(test_movies) if self.use_regression_output else [None] * len(test_movies)
        
        correct = sum(1 for pred, true in zip(predictions, test_labels) if pred == true)
        accuracy = correct / len(test_labels)
        
        if verbose:
            print("=== NEURAL NETWORK EVALUATION RESULTS ===")
            for i, (movie_data, true_label, pred_label, prob, score) in enumerate(zip(
                test_movies, test_labels, predictions, probabilities, success_scores)):
                status = "✓" if pred_label == true_label else "✗"
                score_str = f", Score: {score:.1f}" if score is not None else ""
                print(f"{status} '{movie_data['Title']}' -> Predicted: {pred_label} ({prob:.3f}{score_str}), Actual: {true_label}")
            
            print(f"\nAccuracy: {correct}/{len(test_labels)} = {accuracy:.3f} ({accuracy*100:.1f}%)")
            
            # Confusion matrix and classification report
            print("\nConfusion Matrix:")
            cm = confusion_matrix(test_labels, predictions, labels=["Success", "Failure"])
            print(pd.DataFrame(cm, index=["Actual Success", "Actual Failure"], 
                             columns=["Predicted Success", "Predicted Failure"]))
            
            print("\nClassification Report:")
            print(classification_report(test_labels, predictions, digits=3))
            
            # Regression metrics if available
            if self.use_regression_output and test_revenues is not None:
                from sklearn.metrics import r2_score, mean_absolute_error
                r2 = r2_score(test_revenues, success_scores)
                mae = mean_absolute_error(test_revenues, success_scores)
                print(f"\nRegression Metrics (Success Score vs Actual Revenue):")
                print(f"R² Score: {r2:.3f}")
                print(f"MAE: {mae:.3f} million")
        
        return accuracy, predictions, success_scores
    
    def plot_training_history(self):
        """Plot training loss and accuracy"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Loss plot
            ax1.plot(self.training_history['loss'], 'b-', linewidth=2, label='Train')
            if 'val_loss' in self.training_history:
                ax1.plot(self.training_history['val_loss'], 'r-', linewidth=2, label='Validation')
            ax1.set_title('Training Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Accuracy plot
            ax2.plot([acc * 100 for acc in self.training_history['accuracy']], 'g-', linewidth=2, label='Train')
            if 'val_accuracy' in self.training_history:
                ax2.plot([acc * 100 for acc in self.training_history['val_accuracy']], 'm-', linewidth=2, label='Validation')
            ax2.set_title('Training Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('270nn_training_history.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("Training history saved as 'nn_training_history.png'")
        except Exception as e:
            print(f"Could not create training history plot: {e}")


def load_movie_data(filename):
    """Load movie data from CSV file"""
    movies = []
    with open(filename, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                row['Year'] = int(row['Year'])
                row['Runtime (min)'] = int(row['Runtime (min)'])
                
                revenue = row['Gross Revenue (million)']
                row['Gross Revenue (million)'] = float(revenue) if revenue else 0.0
                
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
    """Calculate success labels and revenues from the dataset"""
    labels = []
    revenues = []
    
    if 'Success' in movies[0]:
        print("Using existing Success labels from dataset")
        for movie in movies:
            success_value = int(movie['Success'])
            if success_value == 1:
                labels.append("Success")
            else:
                labels.append("Failure")
            revenues.append(float(movie['Gross Revenue (million)']))
    else:
        # Fallback to revenue-based calculation
        revenues = [movie["Gross Revenue (million)"] for movie in movies]
        average_revenue = sum(revenues) / len(revenues)
        
        for movie in movies:
            if movie["Gross Revenue (million)"] > average_revenue:
                labels.append("Success")
            else:
                labels.append("Failure")
    
    return labels, revenues

def random_split(movies, labels, revenues=None, train_ratio=0.7, random_seed=42):
    """Split data into training and test sets"""
    random.seed(random_seed)
    
    indices = list(range(len(movies)))
    random.shuffle(indices)
    
    split_point = int(train_ratio * len(movies))
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]
    
    train_movies = [movies[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    train_revenues = [revenues[i] for i in train_indices] if revenues is not None else None
    
    test_movies = [movies[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    test_revenues = [revenues[i] for i in test_indices] if revenues is not None else None
    
    return train_movies, train_labels, train_revenues, test_movies, test_labels, test_revenues

def compare_architectures(movies, labels, revenues):
    """Compare different neural network architectures"""
    print("\n" + "="*60)
    print("NEURAL NETWORK ARCHITECTURE COMPARISON")
    print("="*60)
    
    architectures = [
        ([32], "Single Hidden Layer (32)"),
        ([64], "Single Hidden Layer (64)"),
        ([32, 16], "Two Hidden Layers (32, 16)"),
        ([64, 32], "Two Hidden Layers (64, 32)"),
        ([128, 64, 32], "Three Hidden Layers (128, 64, 32)")
    ]
    
    results = []
    
    for hidden_sizes, description in architectures:
        print(f"\nTesting: {description}")
        
        # Split data
        train_movies, train_labels, train_revenues, test_movies, test_labels, test_revenues = random_split(
            movies, labels, revenues
        )
        
        # Create and train classifier
        nn_classifier = NeuralNetworkClassifier(
            hidden_sizes=hidden_sizes,
            learning_rate=0.001,
            epochs=50,
            batch_size=16,
            dropout_rate=0.2,
            use_regression_output=True
        )
        
        nn_classifier.fit(train_movies, train_labels, train_revenues)
        accuracy, _, scores = nn_classifier.evaluate(test_movies, test_labels, test_revenues)
        
        results.append((description, accuracy, hidden_sizes))
        print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Find best architecture
    best_result = max(results, key=lambda x: x[1])
    print(f"\n" + "="*40)
    print(f"BEST ARCHITECTURE:")
    print(f"Architecture: {best_result[0]}")
    print(f"Accuracy: {best_result[1]:.3f} ({best_result[1]*100:.1f}%)")
    print(f"Hidden sizes: {best_result[2]}")
    print("="*40)
    
    return best_result[2]

def main():
    print("=== NEURAL NETWORK MOVIE SUCCESS PREDICTION ===")
    
    # Load data
    input_filename = '30movies_with_success.csv'
    
    try:
        movie_dataset = load_movie_data(input_filename)
        print(f"Loaded {len(movie_dataset)} movies from {input_filename}")
    except FileNotFoundError:
        print(f"Error: File '{input_filename}' not found.")
        return
    
    # Calculate labels and revenues
    labels, revenues = calculate_success_labels(movie_dataset)
    
    # Compare different architectures
    best_hidden_sizes = compare_architectures(movie_dataset, labels, revenues)
    
    # Train final model with best architecture
    print(f"\n" + "="*60)
    print("FINAL MODEL TRAINING")
    print("="*60)
    
    # Split data with validation set
    train_movies, train_labels, train_revenues, temp_movies, temp_labels, temp_revenues = random_split(
        movie_dataset, labels, revenues, train_ratio=0.6
    )
    val_movies, val_labels, val_revenues, test_movies, test_labels, test_revenues = random_split(
        temp_movies, temp_labels, temp_revenues, train_ratio=0.5
    )
    
    print(f"Training set: {len(train_movies)} movies")
    print(f"Validation set: {len(val_movies)} movies")
    print(f"Test set: {len(test_movies)} movies")
    
    # Create final classifier
    final_nn = NeuralNetworkClassifier(
        hidden_sizes=best_hidden_sizes,
        learning_rate=0.001,
        epochs=100,
        batch_size=16,
        dropout_rate=0.2,
        use_feature_selection=True,
        use_regression_output=True
    )
    
    # Train with validation
    final_nn.fit(
        train_movies, train_labels, train_revenues,
        validation_data=(val_movies, val_labels, val_revenues)
    )
    
    # Plot training history
    final_nn.plot_training_history()
    
    # Final evaluation
    print(f"\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    accuracy, predictions, success_scores = final_nn.evaluate(
        test_movies, test_labels, test_revenues, verbose=True
    )
    
    # Rank movies by predicted success score
    ranked_movies = sorted(
        zip(test_movies, success_scores, predictions),
        key=lambda x: x[1],  # Sort by success score
        reverse=True
    )
    
    print("\nTOP 5 MOST SUCCESSFUL MOVIES (PREDICTED):")
    print("=" * 60)
    for i, (movie, score, pred) in enumerate(ranked_movies[:5]):
        actual_rev = movie['Gross Revenue (million)']
        print(f"{i+1}. {movie['Title']} ({movie['Year']})")
        print(f"   Predicted Score: {score:.1f}, Actual Revenue: {actual_rev:.1f}M")
        print(f"   Genre: {movie['Genre']}, Director: {movie['Director']}")
        print(f"   Predicted: {pred}, Actual: {'Success' if pred == test_labels[i] else 'Failure'}")
        print("-" * 60)
    
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Best architecture: {best_hidden_sizes}")
    print(f"Final test accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"Features used: {final_nn.feature_names[:10]}..." if len(final_nn.feature_names) > 10 else f"Features used: {final_nn.feature_names}")
    print(f"Total features: {len(final_nn.feature_names)}")

if __name__ == "__main__":
    main()