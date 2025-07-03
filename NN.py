import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from math import pi

# =============================================
# 1. DATA PREPARATION
# =============================================

metrics = {
    'Naive Bayes': {
        'Accuracy': 0.786,
        'Precision': 1.000,
        'Recall': 0.538,
        'F1': 0.700,
        'Training Time': 2.1,
        'Prediction Time': 0.8,
        'Confusion Matrix': [[15, 0], [6, 7]],
        'Best Features': ['title_length', 'director', 'year_category'],
        'Dataset Sizes': [30, 90, 270],
        'Accuracies': [0.889, 0.786, 0.827]
    },
    'Decision Tree': {
        'Accuracy': 0.878,
        'Precision': 0.750,
        'Recall': 0.500,
        'F1': 0.600,
        'Training Time': 5.3,
        'Prediction Time': 0.5,
        'Confusion Matrix': [[11, 1], [3, 3]],
        'Best Features': ['genre', 'runtime', 'director'],
        'Dataset Sizes': [30, 90, 270],
        'Accuracies': [0.670, 0.878, 0.793]
    },
    'Neural Network': {
        'Accuracy': 0.628,
        'Precision': 1.000,
        'Recall': 0.250,
        'F1': 0.400,
        'Training Time': 32.7,
        'Prediction Time': 1.2,
        'Confusion Matrix': [[10, 0], [6, 2]],
        'Best Features': ['genre', 'director', 'production_company'],
        'Dataset Sizes': [30, 90, 270],
        'Accuracies': [0.556, 0.628, 0.711]
    },
    'SVM': {
        'Accuracy': 0.739,
        'Precision': 0.857,
        'Recall': 0.545,
        'F1': 0.667,
        'Training Time': 12.4,
        'Prediction Time': 1.0,
        'Confusion Matrix': [[11, 1], [5, 6]],
        'Best Features': ['director', 'has_colon', 'genre'],
        'Dataset Sizes': [30, 90, 270],
        'Accuracies': [0.475, 0.739, 0.778]
    }
}

# Fix inconsistencies in data keys
for clf in metrics:
    if 'Confusion Matrix' not in metrics[clf]:
        metrics[clf]['Confusion Matrix'] = metrics[clf].pop('Confusion Matrix', 
                                                          metrics[clf].pop('Confusion Matrix', [[0,0],[0,0]]))

# =============================================
# 2. PLOT STYLE SETUP
# =============================================

# Use a different built-in style instead of seaborn
plt.style.use('default')  # Changed from 'seaborn' to 'default'

# Apply seaborn-like styling manually
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "figure.autolayout": True,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "#F8F8F8"
})

# Color palette
colors = {
    'Naive Bayes': '#4E79A7',
    'Decision Tree': '#F28E2B',
    'Neural Network': '#E15759',
    'SVM': '#59A14F'
}

# =============================================
# 3. PLOT 1: ACCURACY VS DATASET SIZE (DOTTED)
# =============================================

fig, ax = plt.subplots(figsize=(10, 6))
markers = ['o', 's', 'D', '^']

# Background styling
ax.set_facecolor('#F5F5F5')
ax.grid(True, linestyle='--', alpha=0.6)

# Reference lines
for y in np.arange(0.4, 1.01, 0.1):
    ax.axhline(y=y, color='lightgray', linestyle='--', linewidth=0.5)

# Plot each classifier
for i, (clf, data) in enumerate(metrics.items()):
    ax.plot(data['Dataset Sizes'], data['Accuracies'],
            color=colors[clf],
            marker=markers[i],
            linestyle=':',
            linewidth=2,
            markersize=8,
            label=f"{clf} (Max: {max(data['Accuracies']):.1%})")
    
    # Annotate each point
    for x, y in zip(data['Dataset Sizes'], data['Accuracies']):
        ax.text(x, y+0.02, f"{y:.1%}",
                ha='center', va='bottom',
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

ax.set_title("Classifier Accuracy vs Training Set Size\n(Dotted Lines Show Trends)", pad=15)
ax.set_xlabel("Number of Movies in Dataset")
ax.set_ylabel("Accuracy")
ax.set_xticks([30, 90, 270])
ax.set_ylim(0.4, 1.05)
ax.legend(title="Models (Peak Accuracy)", frameon=True, facecolor='white')
plt.savefig("accuracy_vs_size_dotted.png", dpi=300, bbox_inches='tight')
plt.show()

# =============================================
# 4. PLOT 2: RADAR CHART OF PERFORMANCE METRICS
# =============================================

categories = ['Accuracy', 'Precision', 'Recall', 'F1']
N = len(categories)
angles = [n / N * 2 * pi for n in range(N)]
angles += angles[:1]  # Close the loop

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# Draw axis lines
plt.xticks(angles[:-1], categories)
ax.set_rlabel_position(0)
plt.ylim(0, 1.1)

# Draw ytick labels
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], 
           ["20%", "40%", "60%", "80%", "100%"], 
           color="grey", size=10)

# Plot each classifier
for clf, data in metrics.items():
    values = [
        data['Accuracy'],
        data['Precision'],
        data['Recall'],
        data['F1']
    ]
    values += values[:1]  # Close the loop
    ax.plot(angles, values, linewidth=2, linestyle='solid', 
            label=clf, color=colors[clf])
    ax.fill(angles, values, color=colors[clf], alpha=0.1)

# Add legend and title
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
plt.title("Classifier Performance Radar Chart\n(All Metrics Normalized)", pad=20)
plt.savefig("performance_radar_chart.png", dpi=300, bbox_inches='tight')
plt.show()

# =============================================
# 5. PLOT 3: TIME PERFORMANCE (STACKED BAR)
# =============================================

fig, ax = plt.subplots(figsize=(10, 5))

# Prepare time data
classifiers = list(metrics.keys())
training_times = [metrics[clf]['Training Time'] for clf in classifiers]
prediction_times = [metrics[clf]['Prediction Time'] for clf in classifiers]

# Plot stacked bars
ax.bar(classifiers, training_times, label='Training Time', 
       color=[colors[clf] for clf in classifiers], alpha=0.7)
ax.bar(classifiers, prediction_times, bottom=training_times, 
       label='Prediction Time', color=[colors[clf] for clf in classifiers], alpha=0.4)

# Add value labels
for i, clf in enumerate(classifiers):
    total_time = training_times[i] + prediction_times[i]
    ax.text(i, total_time + 1, f"{total_time:.1f}s", 
            ha='center', va='bottom', fontsize=10)

ax.set_title("Training vs Prediction Time by Classifier")
ax.set_ylabel("Time (seconds)")
ax.legend()
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.savefig("time_performance_stacked.png", dpi=300, bbox_inches='tight')
plt.show()

# =============================================
# 6. PLOT 4: CONFUSION MATRICES (NORMALIZED)
# =============================================

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

for i, (clf, data) in enumerate(metrics.items()):
    cm = np.array(data['Confusion Matrix'])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                display_labels=['Failure', 'Success'])
    disp.plot(ax=axs[i], cmap='Blues', colorbar=False)
    
    # Add normalized values
    for j in range(2):
        for k in range(2):
            axs[i].text(k, j, 
                       f"{cm[j,k]}\n({cm_norm[j,k]:.1%})", 
                       ha='center', va='center',
                       color='white' if cm[j,k] > cm.max()/2 else 'black')
    
    axs[i].set_title(f"{clf}\nAccuracy: {data['Accuracy']:.1%}")
    axs[i].grid(False)

plt.tight_layout()
plt.savefig("confusion_matrices_normalized.png", dpi=300, bbox_inches='tight')
plt.show()

# =============================================
# 7. PLOT 5: FEATURE IMPORTANCE (STACKED)
# =============================================

# Create feature importance dataframe
all_features = set()
for data in metrics.values():
    all_features.update(data['Best Features'])

# Convert to sorted list for consistent indexing
all_features = sorted(list(all_features))
    
feature_importance = pd.DataFrame(
    {clf: [1 if feat in metrics[clf]['Best Features'] else 0 
           for feat in all_features]
     for clf in metrics.keys()},
    index=all_features
).T

# Sort by total importance
feature_importance = feature_importance.loc[:, feature_importance.sum().sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(10, 5))
feature_importance.plot(kind='barh', stacked=True, ax=ax, 
                        color=[colors[clf] for clf in metrics.keys()])

ax.set_title("Feature Importance Across Classifiers")
ax.set_xlabel("Number of Models Using Feature")
ax.legend(title="Classifiers", bbox_to_anchor=(1.05, 1))
plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.savefig("feature_importance_stacked.png", dpi=300, bbox_inches='tight')
plt.show()

# =============================================
# 8. PLOT 6: SIDE-BY-SIDE METRICS COMPARISON
# =============================================

metrics_to_compare = ['Accuracy', 'Precision', 'Recall', 'F1']
x = np.arange(len(metrics_to_compare))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))

for i, clf in enumerate(metrics.keys()):
    values = [metrics[clf][m] for m in metrics_to_compare]
    ax.bar(x + i*width, values, width, label=clf, color=colors[clf])
    
    # Add value labels
    for j, val in enumerate(values):
        ax.text(x[j] + i*width, val + 0.02, f"{val:.3f}", 
                ha='center', va='bottom', fontsize=9)

ax.set_title("Detailed Performance Metrics by Classifier")
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(metrics_to_compare)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.1)
ax.legend()
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.savefig("detailed_metrics_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

print("All plots have been generated successfully!")