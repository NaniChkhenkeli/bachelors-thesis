from collections import Counter, defaultdict

# Dataset
data = [
    {"Gender": "M", "Height": "Medium", "Weight": "Normal", "Hair": "Short"},
    {"Gender": "F", "Height": "Small", "Weight": "Light", "Hair": "Long"},
    {"Gender": "M", "Height": "Tall", "Weight": "Heavy", "Hair": "Short"},
    {"Gender": "F", "Height": "Small", "Weight": "Normal", "Hair": "Long"},
    {"Gender": "F", "Height": "Tall", "Weight": "Normal", "Hair": "Long"},
    {"Gender": "F", "Height": "Small", "Weight": "Light", "Hair": "Short"},
    {"Gender": "M", "Height": "Small", "Weight": "Heavy", "Hair": "Short"},
    {"Gender": "F", "Height": "Medium", "Weight": "Normal", "Hair": "Short"},
    {"Gender": "F", "Height": "Medium", "Weight": "Light", "Hair": "Long"},
    {"Gender": "M", "Height": "Tall", "Weight": "Normal", "Hair": "Short"},
]

target_col = "Gender"
features = [key for key in data[0] if key != target_col]

# Get class counts
class_counts = Counter(row[target_col] for row in data)
total_samples = len(data)

# Get all possible values for each feature
feature_values = defaultdict(set)
for row in data:
    for feature in features:
        feature_values[feature].add(row[feature])

# Count feature values per class
feature_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
for row in data:
    cls = row[target_col]
    for feature in features:
        value = row[feature]
        feature_counts[cls][feature][value] += 1

# Person to classify
sample = {"Height": "Tall", "Weight": "Light", "Hair": "Long"}

# Compute posteriors with Laplace smoothing
print("Detailed Computation:")
best_class = None
best_prob = 0
for cls in class_counts:
    prior = class_counts[cls] / total_samples
    prob = prior
    print(f"\nFor class {cls}:")
    print(f"P({cls}) = {class_counts[cls]}/{total_samples} = {prior:.3f}")
    
    numerator_expr = [str(class_counts[cls])]
    denominator_expr = [str(total_samples)]

    for feature in features:
        value = sample[feature]
        count = feature_counts[cls][feature][value]
        total_in_class = class_counts[cls]
        num_values = len(feature_values[feature])
        
        # Laplace smoothing
        smoothed_count = count + 1
        smoothed_total = total_in_class + num_values
        
        prob *= smoothed_count / smoothed_total

        print(f"P({value}|{cls}) = ({count}+1)/({total_in_class}+{num_values}) = {smoothed_count}/{smoothed_total}")
        numerator_expr.append(str(smoothed_count))
        denominator_expr.append(str(smoothed_total))

    print("Combined (Numerator): " + " * ".join(numerator_expr))
    print("Combined (Denominator): " + " * ".join(denominator_expr))
    print(f"Posterior = {prob:.6f}")

    if prob > best_prob:
        best_prob = prob
        best_class = cls

print("\nConclusion:")
print(f"A person with features {sample} is most likely '{best_class.upper()}' with probability {best_prob:.6f}")
