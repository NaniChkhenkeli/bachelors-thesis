import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

# Final metrics from all classifiers with dataset size information
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
        'Accuracies': [0.833, 0.778, 0.709]
    },
    'Decision Tree': {
        'Accuracy': 0.778,
        'Precision': 0.750,
        'Recall': 0.500,
        'F1': 0.600,
        'Training Time': 5.3,
        'Prediction Time': 0.5,
        'Confusion Matrix': [[11, 1], [3, 3]],
        'Best Features': ['genre', 'runtime', 'director'],
        'Dataset Sizes': [30, 90, 270],
        'Accuracies': [0.600, 0.778, 0.593]
    },
    'Neural Network': {
        'Accuracy': 0.667,
        'Precision': 1.000,
        'Recall': 0.250,
        'F1': 0.400,
        'Training Time': 32.7,
        'Prediction Time': 1.2,
        'Confusion Matrix': [[10, 0], [6, 2]],
        'Best Features': ['genre', 'director', 'production_company'],
        'Dataset Sizes': [30, 90, 270],
        'Accuracies': [0.500, 0.667, 0.611]
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
        'Accuracies': [0.250, 0.739, 0.618]
    }
}

# Set plot style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150
})

# 0. Accuracy vs Training Set Size Plot (with dotted lines)
fig, ax = plt.subplots(figsize=(11, 7))
colors = plt.cm.Set2(np.linspace(0, 1, len(metrics)))
markers = ['o', 's', '^', 'D']

# Background and grid
ax.set_facecolor("#f9f9f9")
ax.grid(True, which='major', linestyle='--', linewidth=0.6, color='gray', alpha=0.3)

# Horizontal reference lines
for y in np.arange(0.1, 1.01, 0.1):
    ax.axhline(y=y, linestyle='--', color='lightgray', linewidth=0.5)

# Plot each classifier with dotted lines
for i, (clf, values) in enumerate(metrics.items()):
    x = values['Dataset Sizes']
    y = values['Accuracies']
    ax.plot(x, y,
            label=clf,
            color=colors[i],
            marker=markers[i],
            linestyle=':',  # DOTTED line style
            linewidth=2,
            markersize=9,
            markerfacecolor='white',
            markeredgewidth=2,
            markeredgecolor=colors[i])

    # Annotate accuracy values
    for j in range(len(x)):
        ax.annotate(f"{y[j]:.2f}", (x[j], y[j] + 0.03), ha='center', fontsize=10)

# Labels and title
ax.set_title("📊 Classifier Accuracy vs. Training Set Size", pad=20)
ax.set_xlabel("Training Set Size (movies)")
ax.set_ylabel("Accuracy")
ax.set_xticks([30, 90, 270])
ax.set_ylim(0, 1.05)
ax.legend(title="Classifier", loc="lower right", frameon=True)

plt.tight_layout()
plt.savefig("accuracy_vs_size_dotted.png", dpi=300, bbox_inches="tight")
plt.show()

# 1. All Metrics Curve Plot
plt.figure(figsize=(12, 6))
categories = ['Accuracy', 'Precision', 'Recall', 'F1']
x = np.arange(len(categories))

# Create a color palette
colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))

for (clf, values), color in zip(metrics.items(), colors):
    stats = [
        values['Accuracy'],
        values['Precision'],
        values['Recall'],
        values['F1']
    ]
    plt.plot(x, stats, 'o-', color=color, linewidth=2, markersize=8, label=clf)
    # Add data labels
    for i, val in enumerate(stats):
        plt.text(x[i], val+0.02, f"{val:.3f}", ha='center', va='bottom', fontsize=9)

plt.title('Classifier Performance Comparison', size=14, pad=20)
plt.xticks(x, categories)
plt.ylabel('Score')
plt.ylim(0, 1.1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('curve_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Time Performance Curve Plot
plt.figure(figsize=(12, 6))
time_metrics = ['Training Time', 'Prediction Time']

for i, (clf, values) in enumerate(metrics.items()):
    times = [values['Training Time'], values['Prediction Time']]
    plt.plot(time_metrics, times, 'o-', color=colors[i], linewidth=2, markersize=8, label=clf)
    # Add data labels
    for j, val in enumerate(times):
        plt.text(j, val+0.5, f"{val:.1f}s", ha='center', va='bottom', fontsize=9)

plt.title('Classifier Time Performance Comparison', size=14, pad=20)
plt.ylabel('Time (seconds)')
plt.ylim(0, max([v['Training Time'] for v in metrics.values()]) + 5)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('time_curve_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Side-by-Score Metrics Comparison
metrics_df = pd.DataFrame(metrics).T
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1']

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(metrics_df))
width = 0.2

for i, metric in enumerate(metrics_to_plot):
    ax.bar(x + i*width, metrics_df[metric], width, label=metric)

ax.set_title('Classifier Performance Metrics Comparison', fontsize=14)
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(metrics_df.index)
ax.set_ylabel('Score', fontsize=12)
ax.set_ylim(0, 1.1)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('all_metrics_comparison.png', dpi=300)
plt.show()

# 4. Confusion Matrix Heatmaps
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.ravel()

for i, (clf, values) in enumerate(metrics.items()):
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(values['Confusion Matrix']), 
                                display_labels=['Failure', 'Success'])
    disp.plot(ax=axs[i], cmap='Blues', colorbar=False)
    axs[i].set_title(f'{clf} Confusion Matrix')
    axs[i].grid(False)
    
    # Add accuracy annotation
    axs[i].text(0.5, -0.15, f"Accuracy: {values['Accuracy']:.1%}", 
               ha='center', va='center', transform=axs[i].transAxes)

plt.tight_layout()
plt.savefig('confusion_matrices_comparison.png', dpi=300)
plt.show()

# 5. Feature Importance Comparison
feature_importance = {}
for clf, values in metrics.items():
    if 'Best Features' in values:
        for feature in values['Best Features']:
            if feature not in feature_importance:
                feature_importance[feature] = 0
            feature_importance[feature] += 1

features = list(feature_importance.keys())
counts = list(feature_importance.values())

plt.figure(figsize=(10, 6))
plt.barh(features, counts, color='#55A868')
plt.title('Most Important Features Across All Classifiers', fontsize=14)
plt.xlabel('Number of Models Using Feature', fontsize=12)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance_comparison.png', dpi=300)
plt.show()