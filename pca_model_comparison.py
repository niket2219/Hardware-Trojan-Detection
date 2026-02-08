"""
PCA-Based Hardware Trojan Detection: Model Comparison
This script applies PCA dimensionality reduction and compares 7 ML models
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess_data import prepare_data

# Import ML models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


# ============================================================================
# 1. PCA Analysis Function
# ============================================================================

def apply_pca_analysis(x_train, x_test, n_components=None):
    """
    Apply PCA and return reduced features with explained variance info

    Args:
        x_train: Training features
        x_test: Test features
        n_components: Number of components (if None, use 20)

    Returns:
        x_train_pca: PCA-transformed training features
        x_test_pca: PCA-transformed test features
        pca_model: Fitted PCA model
        n_components_used: Actual number of components used
    """

    if n_components is None:
        n_components = min(20, x_train.shape[1])

    print(f"\n{'='*70}")
    print(f"PCA ANALYSIS")
    print(f"{'='*70}")
    print(f"Original dimensions: {x_train.shape[1]}")
    print(f"Applying PCA with {n_components} components...")

    # Fit PCA on training data
    pca = PCA(n_components=n_components, random_state=42)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)

    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    print(f"\nReduced dimensions: {x_train_pca.shape[1]}")
    print(
        f"Dimension reduction: {(1 - x_train_pca.shape[1]/x_train.shape[1])*100:.1f}%")

    # Print variance explained
    print(f"\n{'PC':<5} {'Variance':<12} {'Cumulative':<12}")
    print("-" * 30)
    for i in range(min(10, n_components)):
        print(
            f"{i+1:<5} {explained_variance[i]:.4f}      {cumulative_variance[i]:.4f}")

    if n_components > 10:
        print("...")

    print(
        f"\nTotal variance explained by {n_components} components: {cumulative_variance[-1]:.4f} ({cumulative_variance[-1]*100:.2f}%)")
    print(f"Information retained: {cumulative_variance[-1]*100:.2f}%")
    print(f"Information lost: {(1-cumulative_variance[-1])*100:.2f}%")

    return x_train_pca, x_test_pca, pca, n_components


# ============================================================================
# 2. Model Training Functions
# ============================================================================

def train_svm(x_train, x_test, y_train, y_test):
    """Support Vector Machine"""
    print("\n" + "="*70)
    print("TRAINING: Support Vector Machine (SVM)")
    print("="*70)

    start = time.time()
    clf = SVC(kernel="rbf", C=10, gamma=1, probability=True)
    clf.fit(x_train, y_train.ravel())
    train_time = time.time() - start

    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0) * 100

    # Calculate AUC if binary classification
    try:
        y_pred_proba = clf.predict_proba(x_test)
        if y_pred_proba.shape[1] == 2:
            auc = roc_auc_score(y_test, y_pred_proba[:, 1]) * 100
        else:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr') * 100
    except:
        auc = np.nan

    print(f"Training Time: {train_time:.4f} seconds")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"F1-Score: {f1:.2f}%")
    print(f"ROC-AUC: {auc:.2f}%" if not np.isnan(auc) else "ROC-AUC: N/A")

    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    return {
        'name': 'SVM',
        'model': clf,
        'accuracy': accuracy,
        'f1_score': f1,
        'auc': auc,
        'time': train_time,
        'y_pred': y_pred,
        'cm': cm
    }


def train_random_forest(x_train, x_test, y_train, y_test):
    """Random Forest"""
    print("\n" + "="*70)
    print("TRAINING: Random Forest (RF)")
    print("="*70)

    start = time.time()
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    clf.fit(x_train, y_train.ravel())
    train_time = time.time() - start

    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0) * 100

    # Calculate AUC
    try:
        y_pred_proba = clf.predict_proba(x_test)
        if y_pred_proba.shape[1] == 2:
            auc = roc_auc_score(y_test, y_pred_proba[:, 1]) * 100
        else:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr') * 100
    except:
        auc = np.nan

    print(f"Training Time: {train_time:.4f} seconds")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"F1-Score: {f1:.2f}%")
    print(f"ROC-AUC: {auc:.2f}%" if not np.isnan(auc) else "ROC-AUC: N/A")

    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    return {
        'name': 'Random Forest',
        'model': clf,
        'accuracy': accuracy,
        'f1_score': f1,
        'auc': auc,
        'time': train_time,
        'y_pred': y_pred,
        'cm': cm
    }


def train_gradient_boosting(x_train, x_test, y_train, y_test):
    """Gradient Boosting"""
    print("\n" + "="*70)
    print("TRAINING: Gradient Boosting (GB)")
    print("="*70)

    start = time.time()
    clf = GradientBoostingClassifier(
        learning_rate=0.1, n_estimators=100, random_state=42)
    clf.fit(x_train, y_train.ravel())
    train_time = time.time() - start

    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0) * 100

    # Calculate AUC
    try:
        y_pred_proba = clf.predict_proba(x_test)
        if y_pred_proba.shape[1] == 2:
            auc = roc_auc_score(y_test, y_pred_proba[:, 1]) * 100
        else:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr') * 100
    except:
        auc = np.nan

    print(f"Training Time: {train_time:.4f} seconds")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"F1-Score: {f1:.2f}%")
    print(f"ROC-AUC: {auc:.2f}%" if not np.isnan(auc) else "ROC-AUC: N/A")

    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    return {
        'name': 'Gradient Boosting',
        'model': clf,
        'accuracy': accuracy,
        'f1_score': f1,
        'auc': auc,
        'time': train_time,
        'y_pred': y_pred,
        'cm': cm
    }


def train_xgboost(x_train, x_test, y_train, y_test):
    """XGBoost"""
    print("\n" + "="*70)
    print("TRAINING: XGBoost")
    print("="*70)

    start = time.time()
    clf = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss',
                        random_state=42, n_jobs=-1)
    clf.fit(x_train, y_train.ravel(), verbose=False)
    train_time = time.time() - start

    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0) * 100

    # Calculate AUC
    try:
        y_pred_proba = clf.predict_proba(x_test)
        if y_pred_proba.shape[1] == 2:
            auc = roc_auc_score(y_test, y_pred_proba[:, 1]) * 100
        else:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr') * 100
    except:
        auc = np.nan

    print(f"Training Time: {train_time:.4f} seconds")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"F1-Score: {f1:.2f}%")
    print(f"ROC-AUC: {auc:.2f}%" if not np.isnan(auc) else "ROC-AUC: N/A")

    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    return {
        'name': 'XGBoost',
        'model': clf,
        'accuracy': accuracy,
        'f1_score': f1,
        'auc': auc,
        'time': train_time,
        'y_pred': y_pred,
        'cm': cm
    }


def train_knn(x_train, x_test, y_train, y_test):
    """K-Nearest Neighbors"""
    print("\n" + "="*70)
    print("TRAINING: K-Nearest Neighbors (KNN)")
    print("="*70)

    start = time.time()
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(x_train, y_train.ravel())
    train_time = time.time() - start

    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0) * 100

    # Calculate AUC
    try:
        y_pred_proba = clf.predict_proba(x_test)
        if y_pred_proba.shape[1] == 2:
            auc = roc_auc_score(y_test, y_pred_proba[:, 1]) * 100
        else:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr') * 100
    except:
        auc = np.nan

    print(f"Training Time: {train_time:.4f} seconds")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"F1-Score: {f1:.2f}%")
    print(f"ROC-AUC: {auc:.2f}%" if not np.isnan(auc) else "ROC-AUC: N/A")

    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    return {
        'name': 'K-Nearest Neighbors',
        'model': clf,
        'accuracy': accuracy,
        'f1_score': f1,
        'auc': auc,
        'time': train_time,
        'y_pred': y_pred,
        'cm': cm
    }


def train_logistic_regression(x_train, x_test, y_train, y_test):
    """Logistic Regression"""
    print("\n" + "="*70)
    print("TRAINING: Logistic Regression")
    print("="*70)

    start = time.time()
    clf = LogisticRegression(
        random_state=42, solver='lbfgs', max_iter=500, multi_class='multinomial')
    clf.fit(x_train, y_train.ravel())
    train_time = time.time() - start

    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0) * 100

    # Calculate AUC
    try:
        y_pred_proba = clf.predict_proba(x_test)
        if y_pred_proba.shape[1] == 2:
            auc = roc_auc_score(y_test, y_pred_proba[:, 1]) * 100
        else:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr') * 100
    except:
        auc = np.nan

    print(f"Training Time: {train_time:.4f} seconds")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"F1-Score: {f1:.2f}%")
    print(f"ROC-AUC: {auc:.2f}%" if not np.isnan(auc) else "ROC-AUC: N/A")

    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    return {
        'name': 'Logistic Regression',
        'model': clf,
        'accuracy': accuracy,
        'f1_score': f1,
        'auc': auc,
        'time': train_time,
        'y_pred': y_pred,
        'cm': cm
    }


def train_mlp(x_train, x_test, y_train, y_test):
    """Multilayer Perceptron"""
    print("\n" + "="*70)
    print("TRAINING: Multilayer Perceptron (MLP)")
    print("="*70)

    # Prepare data for neural network
    y_train_categorical = to_categorical(y_train.ravel())
    y_test_categorical = to_categorical(y_test.ravel())

    num_classes = y_train_categorical.shape[1]

    # Build model
    model = Sequential()
    model.add(Dense(32, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    start = time.time()
    model.fit(x_train, y_train_categorical, epochs=50, batch_size=10,
              validation_split=0.2, verbose=0)
    train_time = time.time() - start

    y_pred_proba = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    accuracy = accuracy_score(y_test.ravel(), y_pred) * 100
    f1 = f1_score(y_test.ravel(), y_pred,
                  average='macro', zero_division=0) * 100

    # Calculate AUC
    try:
        if y_pred_proba.shape[1] == 2:
            auc = roc_auc_score(y_test.ravel(), y_pred_proba[:, 1]) * 100
        else:
            auc = roc_auc_score(y_test.ravel(), y_pred_proba,
                                multi_class='ovr') * 100
    except:
        auc = np.nan

    print(f"Training Time: {train_time:.4f} seconds")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"F1-Score: {f1:.2f}%")
    print(f"ROC-AUC: {auc:.2f}%" if not np.isnan(auc) else "ROC-AUC: N/A")

    cm = confusion_matrix(y_test.ravel(), y_pred)
    print(f"Confusion Matrix:\n{cm}")

    return {
        'name': 'MLP',
        'model': model,
        'accuracy': accuracy,
        'f1_score': f1,
        'auc': auc,
        'time': train_time,
        'y_pred': y_pred,
        'cm': cm
    }


# ============================================================================
# 3. Visualization Functions
# ============================================================================

def plot_accuracy_comparison(results):
    """Plot accuracy comparison across models"""
    names = [r['name'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    f1_scores = [r['f1_score'] for r in results]
    times = [r['time'] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Accuracy plot
    colors = ['green' if acc >= 94 else 'orange' if acc >=
              90 else 'red' for acc in accuracies]
    axes[0].bar(names, accuracies, color=colors,
                edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Model Accuracy Comparison (PCA Features)',
                      fontsize=13, fontweight='bold')
    axes[0].set_ylim([80, 100])
    axes[0].axhline(y=90, color='r', linestyle='--', label='90% threshold')
    axes[0].tick_params(axis='x', rotation=45)
    for i, acc in enumerate(accuracies):
        axes[0].text(i, acc + 0.5, f'{acc:.2f}%',
                     ha='center', fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # F1-Score plot
    axes[1].bar(names, f1_scores, color='skyblue',
                edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('F1-Score (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Model F1-Score Comparison (PCA Features)',
                      fontsize=13, fontweight='bold')
    axes[1].set_ylim([80, 100])
    axes[1].tick_params(axis='x', rotation=45)
    for i, f1 in enumerate(f1_scores):
        axes[1].text(i, f1 + 0.5, f'{f1:.2f}%', ha='center', fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    # Training Time plot
    axes[2].bar(names, times, color='coral', edgecolor='black', linewidth=1.5)
    axes[2].set_ylabel('Training Time (seconds)',
                       fontsize=12, fontweight='bold')
    axes[2].set_title(
        'Model Training Time Comparison (PCA Features)', fontsize=13, fontweight='bold')
    axes[2].tick_params(axis='x', rotation=45)
    for i, t in enumerate(times):
        axes[2].text(i, t + 0.05, f'{t:.4f}s',
                     ha='center', fontweight='bold', fontsize=9)
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('pca_model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: pca_model_comparison.png")
    plt.show()


def plot_results_table(results):
    """Create and display results table"""
    data = []
    for r in results:
        data.append({
            'Model': r['name'],
            'Accuracy (%)': f"{r['accuracy']:.2f}",
            'F1-Score (%)': f"{r['f1_score']:.2f}",
            'ROC-AUC (%)': f"{r['auc']:.2f}" if not np.isnan(r['auc']) else "N/A",
            'Time (s)': f"{r['time']:.4f}"
        })

    df = pd.DataFrame(data)

    print("\n" + "="*100)
    print("MODEL COMPARISON TABLE (PCA FEATURES)")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)

    return df


def plot_confusion_matrices(results):
    """Plot confusion matrices for all models"""
    n_models = len(results)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, r in enumerate(results):
        cm = r['cm']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    cbar_kws={'label': 'Count'}, annot_kws={'fontsize': 10})
        axes[idx].set_title(
            f"{r['name']}\nAccuracy: {r['accuracy']:.2f}%", fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig('pca_confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: pca_confusion_matrices.png")
    plt.show()


# ============================================================================
# 4. Main Execution
# ============================================================================

def main():
    """Main execution function"""

    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "PCA-BASED HARDWARE TROJAN DETECTION" + " "*19 + "║")
    print("║" + " "*22 + "Model Comparison Analysis" + " "*22 + "║")
    print("╚" + "="*68 + "╝")

    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    x_train, x_test, y_train, y_test = prepare_data()
    print(f"✓ Data loaded successfully")
    print(f"  Training samples: {x_train.shape[0]}")
    print(f"  Test samples: {x_test.shape[0]}")
    print(f"  Original features: {x_train.shape[1]}")

    # Apply PCA
    x_train_pca, x_test_pca, pca_model, n_comp = apply_pca_analysis(
        x_train, x_test, n_components=20)

    # Train all models
    results = []
    results.append(train_svm(x_train_pca, x_test_pca, y_train, y_test))
    results.append(train_random_forest(
        x_train_pca, x_test_pca, y_train, y_test))
    results.append(train_gradient_boosting(
        x_train_pca, x_test_pca, y_train, y_test))
    results.append(train_xgboost(x_train_pca, x_test_pca, y_train, y_test))
    results.append(train_knn(x_train_pca, x_test_pca, y_train, y_test))
    results.append(train_logistic_regression(
        x_train_pca, x_test_pca, y_train, y_test))
    results.append(train_mlp(x_train_pca, x_test_pca, y_train, y_test))

    # Display results table
    results_df = plot_results_table(results)

    # Identify best model
    best_model = max(results, key=lambda x: x['accuracy'])
    print(f"\n{'='*70}")
    print(
        f"BEST MODEL: {best_model['name']} with {best_model['accuracy']:.2f}% accuracy")
    print(f"{'='*70}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_accuracy_comparison(results)
    plot_confusion_matrices(results)

    # Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    accuracies = [r['accuracy'] for r in results]
    print(f"Mean Accuracy: {np.mean(accuracies):.2f}%")
    print(f"Max Accuracy: {np.max(accuracies):.2f}%")
    print(f"Min Accuracy: {np.min(accuracies):.2f}%")
    print(f"Std Dev: {np.std(accuracies):.2f}%")
    print(
        f"Training Time Range: {np.min([r['time'] for r in results]):.4f}s - {np.max([r['time'] for r in results]):.4f}s")

    print(f"\n✓ PCA Analysis Complete!")
    print(f"\nFiles saved:")
    print(f"  - pca_model_comparison.png")
    print(f"  - pca_confusion_matrices.png")

    return results, results_df


if __name__ == '__main__':
    results, results_df = main()
