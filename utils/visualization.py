import matplotlib.pyplot as plt


def plot_results(results):
    """Plot results from a single model"""
    print(f"\n{'='*50}")
    print(f"Model: {results['name']}")
    print(f"{'='*50}")
    print(f"Accuracy:  {results['accuracy']:.2f}%")
    print(f"F1-Score:  {results['f1']:.2f}%")
    print(f"Precision: {results['precision']:.2f}%")
    print(f"Recall:    {results['recall']:.2f}%")
    print(f"Time:      {results['time']:.4f}s")
    print(f"{'='*50}\n")


def plot_comparison(results_list):
    """Compare multiple model results"""
    models = [r['name'] for r in results_list]
    accuracies = [r['accuracy'] for r in results_list]
    times = [r['time'] for r in results_list]

    # Plot accuracy comparison
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.bar(range(len(models)), accuracies, color='steelblue')
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 105)

    # Plot time comparison
    plt.subplot(1, 2, 2)
    plt.bar(range(len(models)), times, color='coral')
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time Comparison')

    plt.tight_layout()
    plt.savefig('outputs/model_comparison.png', dpi=100, bbox_inches='tight')
    plt.show()

    # Print detailed comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    print(f"{'Model':<25} {'Accuracy':<12} {'F1-Score':<12} {'Precision':<12} {'Recall':<12} {'Time':<10}")
    print("-"*80)
    for r in results_list:
        print(f"{r['name']:<25} {r['accuracy']:<11.2f}% {r['f1']:<11.2f}% {r['precision']:<11.2f}% {r['recall']:<11.2f}% {r['time']:<9.4f}s")
    print("="*80 + "\n")
