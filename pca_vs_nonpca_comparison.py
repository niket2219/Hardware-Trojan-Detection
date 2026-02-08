"""
PCA vs Non-PCA Feature Comparison
Directly compares accuracy of models with and without PCA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess_data import prepare_data
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import time


def train_model(model_name, x_train, x_test, y_train, y_test):
    """Train a single model and return metrics"""

    if model_name == 'Random Forest':
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    elif model_name == 'Gradient Boosting':
        clf = GradientBoostingClassifier(
            learning_rate=0.1, n_estimators=100, random_state=42)
    elif model_name == 'XGBoost':
        clf = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss',
                            random_state=42, verbose=False, n_jobs=-1)
    elif model_name == 'SVM':
        clf = SVC(kernel="rbf", C=10, gamma=1, probability=True)
    elif model_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=5)
    elif model_name == 'Logistic Regression':
        clf = LogisticRegression(
            random_state=42, solver='lbfgs', max_iter=500, multi_class='multinomial')
    else:
        raise ValueError(f"Unknown model: {model_name}")

    start = time.time()
    clf.fit(x_train, y_train.ravel())
    train_time = time.time() - start

    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0) * 100

    try:
        y_pred_proba = clf.predict_proba(x_test)
        if y_pred_proba.shape[1] == 2:
            auc = roc_auc_score(y_test, y_pred_proba[:, 1]) * 100
        else:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr') * 100
    except:
        auc = np.nan

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'auc': auc,
        'time': train_time
    }


def main():
    print("\n" + "="*80)
    print("PCA vs NON-PCA FEATURE COMPARISON")
    print("="*80)

    # Load data
    print("\nLoading data...")
    x_train, x_test, y_train, y_test = prepare_data()

    print(f"[OK] Data loaded")
    print(f"  Original features: {x_train.shape[1]}")
    print(f"  Training samples: {x_train.shape[0]}")
    print(f"  Test samples: {x_test.shape[0]}")

    # Apply PCA
    print("\nApplying PCA (20 components)...")
    pca = PCA(n_components=20, random_state=42)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)
    print(f"[OK] PCA applied")
    print(
        f"  Variance explained: {pca.explained_variance_ratio_.sum()*100:.2f}%")
    print(f"  Dimension reduction: {(1-20/x_train.shape[1])*100:.1f}%")

    # Models to test
    models = ['Random Forest', 'Gradient Boosting',
              'XGBoost', 'KNN', 'SVM', 'Logistic Regression']

    results = []

    print("\n" + "="*80)
    print("TRAINING MODELS WITH ALL FEATURES (NON-PCA)")
    print("="*80)

    for model_name in models:
        print(f"\nTraining {model_name}...")
        metrics = train_model(model_name, x_train, x_test, y_train, y_test)
        results.append({
            'Model': model_name,
            'Features': 'Non-PCA (25)',
            'Accuracy': metrics['accuracy'],
            'F1-Score': metrics['f1_score'],
            'ROC-AUC': metrics['auc'],
            'Time': metrics['time']
        })
        print(
            f"  [OK] {model_name}: {metrics['accuracy']:.2f}% accuracy, {metrics['time']:.4f}s")

    print("\n" + "="*80)
    print("TRAINING MODELS WITH PCA FEATURES")
    print("="*80)

    for model_name in models:
        print(f"\nTraining {model_name}...")
        metrics = train_model(model_name, x_train_pca,
                              x_test_pca, y_train, y_test)
        results.append({
            'Model': model_name,
            'Features': 'PCA (20)',
            'Accuracy': metrics['accuracy'],
            'F1-Score': metrics['f1_score'],
            'ROC-AUC': metrics['auc'],
            'Time': metrics['time']
        })
        print(
            f"  [OK] {model_name}: {metrics['accuracy']:.2f}% accuracy, {metrics['time']:.4f}s")

    # Create comparison table
    df_results = pd.DataFrame(results)

    print("\n" + "="*100)
    print("COMPREHENSIVE COMPARISON TABLE")
    print("="*100)

    # Pivot for better comparison
    pivot_accuracy = df_results.pivot(
        index='Model', columns='Features', values='Accuracy')
    pivot_f1 = df_results.pivot(
        index='Model', columns='Features', values='F1-Score')
    pivot_time = df_results.pivot(
        index='Model', columns='Features', values='Time')

    print("\nACCURACY COMPARISON (%)")
    print("-" * 60)
    print(pivot_accuracy.to_string())

    print("\n\nF1-SCORE COMPARISON (%)")
    print("-" * 60)
    print(pivot_f1.to_string())

    print("\n\nTRAINING TIME COMPARISON (seconds)")
    print("-" * 60)
    print(pivot_time.to_string())

    # Calculate differences
    print("\n" + "="*100)
    print("DIFFERENCE ANALYSIS (PCA - Non-PCA)")
    print("="*100)

    diff_accuracy = pivot_accuracy['PCA (20)'] - pivot_accuracy['Non-PCA (25)']
    diff_time = pivot_time['PCA (20)'] - pivot_time['Non-PCA (25)']

    print("\nAccuracy Difference (PCA - Non-PCA) - Positive = PCA better:")
    print("-" * 60)
    for model in diff_accuracy.index:
        value = diff_accuracy[model]
        sign = "+" if value > 0 else ""
        symbol = "⬆️ PCA Better" if value > 0 else "⬇️ Worse"
        print(f"{model:25} {sign}{value:7.2f}% {symbol}")

    print("\nTraining Time Difference (PCA - Non-PCA) - Negative = PCA faster:")
    print("-" * 60)
    speedup_pct = ((pivot_time['Non-PCA (25)'] -
                   pivot_time['PCA (20)']) / pivot_time['Non-PCA (25)']) * 100
    for model in diff_time.index:
        value = diff_time[model]
        speedup = speedup_pct[model]
        sign = "-" if value < 0 else "+"
        symbol = f"⚡ {speedup:.1f}% faster" if value < 0 else f"🐢 {speedup:.1f}% slower"
        print(f"{model:25} {sign}{abs(value):7.4f}s {symbol}")

    # Create visualization
    print("\n" + "="*80)
    print("Generating comparison visualization...")
    print("="*80)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Accuracy comparison
    for model in models:
        non_pca = pivot_accuracy.loc[model, 'Non-PCA (25)']
        pca = pivot_accuracy.loc[model, 'PCA (20)']
        x_pos = models.index(model)
        axes[0, 0].bar(x_pos - 0.2, non_pca, 0.4, label='Non-PCA' if model ==
                       models[0] else '', color='steelblue', alpha=0.8)
        axes[0, 0].bar(x_pos + 0.2, pca, 0.4, label='PCA' if model ==
                       models[0] else '', color='coral', alpha=0.8)

    axes[0, 0].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title(
        'Accuracy Comparison: Non-PCA vs PCA Features', fontsize=13, fontweight='bold')
    axes[0, 0].set_xticks(range(len(models)))
    axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 0].set_ylim([60, 105])
    axes[0, 0].legend(loc='lower right', fontsize=11)
    axes[0, 0].grid(axis='y', alpha=0.3)

    # F1-Score comparison
    for model in models:
        non_pca = pivot_f1.loc[model, 'Non-PCA (25)']
        pca = pivot_f1.loc[model, 'PCA (20)']
        x_pos = models.index(model)
        axes[0, 1].bar(x_pos - 0.2, non_pca, 0.4, label='Non-PCA' if model ==
                       models[0] else '', color='steelblue', alpha=0.8)
        axes[0, 1].bar(x_pos + 0.2, pca, 0.4, label='PCA' if model ==
                       models[0] else '', color='coral', alpha=0.8)

    axes[0, 1].set_ylabel('F1-Score (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title(
        'F1-Score Comparison: Non-PCA vs PCA Features', fontsize=13, fontweight='bold')
    axes[0, 1].set_xticks(range(len(models)))
    axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 1].set_ylim([30, 105])
    axes[0, 1].legend(loc='lower right', fontsize=11)
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Training time comparison (log scale)
    for model in models:
        non_pca = pivot_time.loc[model, 'Non-PCA (25)']
        pca = pivot_time.loc[model, 'PCA (20)']
        x_pos = models.index(model)
        axes[1, 0].bar(x_pos - 0.2, non_pca, 0.4, label='Non-PCA' if model ==
                       models[0] else '', color='steelblue', alpha=0.8)
        axes[1, 0].bar(x_pos + 0.2, pca, 0.4, label='PCA' if model ==
                       models[0] else '', color='coral', alpha=0.8)

    axes[1, 0].set_ylabel('Training Time (seconds)',
                          fontsize=12, fontweight='bold')
    axes[1, 0].set_title(
        'Training Time Comparison: Non-PCA vs PCA Features', fontsize=13, fontweight='bold')
    axes[1, 0].set_xticks(range(len(models)))
    axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend(loc='upper left', fontsize=11)
    axes[1, 0].grid(axis='y', alpha=0.3, which='both')

    # Accuracy difference
    colors = ['green' if x > 0 else 'red' for x in diff_accuracy.values]
    axes[1, 1].barh(range(len(diff_accuracy)), diff_accuracy.values,
                    color=colors, alpha=0.8, edgecolor='black')
    axes[1, 1].set_yticks(range(len(diff_accuracy)))
    axes[1, 1].set_yticklabels(diff_accuracy.index)
    axes[1, 1].set_xlabel('Accuracy Difference (%)',
                          fontsize=12, fontweight='bold')
    axes[1, 1].set_title(
        'Accuracy Difference (PCA - Non-PCA) | +Green=Better, -Red=Worse', fontsize=13, fontweight='bold')
    axes[1, 1].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    axes[1, 1].grid(axis='x', alpha=0.3)

    for i, v in enumerate(diff_accuracy.values):
        axes[1, 1].text(v + 0.2 if v > 0 else v - 0.2, i, f'{v:+.2f}%', va='center',
                        ha='left' if v > 0 else 'right', fontweight='bold')

    plt.tight_layout()
    plt.savefig('pca_vs_non_pca_comparison.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: pca_vs_non_pca_comparison.png")
    plt.show()

    # Summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)

    avg_acc_non_pca = pivot_accuracy['Non-PCA (25)'].mean()
    avg_acc_pca = pivot_accuracy['PCA (20)'].mean()
    avg_time_non_pca = pivot_time['Non-PCA (25)'].mean()
    avg_time_pca = pivot_time['PCA (20)'].mean()

    print(f"\nAverage Accuracy:")
    print(f"  Non-PCA (25 features): {avg_acc_non_pca:.2f}%")
    print(f"  PCA (20 components):   {avg_acc_pca:.2f}%")
    print(f"  Difference:            {avg_acc_pca - avg_acc_non_pca:+.2f}%")

    print(f"\nAverage Training Time:")
    print(f"  Non-PCA (25 features): {avg_time_non_pca:.4f}s")
    print(f"  PCA (20 components):   {avg_time_pca:.4f}s")
    print(f"  Speedup:               {(avg_time_non_pca / avg_time_pca):.2f}x")
    print(
        f"  Time Saved:            {(avg_time_non_pca - avg_time_pca)*100:.1f}%")

    print("\n[OK] Analysis complete!")

    return df_results, pivot_accuracy, pivot_f1, pivot_time


if __name__ == '__main__':
    df_results, pivot_acc, pivot_f1, pivot_time = main()
