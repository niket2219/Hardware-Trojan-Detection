import sys
from models import train_rf, train_svm, train_lr, train_knn, train_gb, train_xgb, train_mlp
from utils import plot_results, plot_comparison


def main():
    """Main entry point for training models"""

    if len(sys.argv) < 2:
        print("Usage: python main.py [model_name]")
        print("\nAvailable models:")
        print("  - rf              : Random Forest")
        print("  - svm             : Support Vector Machine")
        print("  - lr              : Logistic Regression")
        print("  - knn             : K-Nearest Neighbors")
        print("  - gb              : Gradient Boosting")
        print("  - xgb             : XGBoost")
        print("  - mlp             : Multilayer Perceptron")
        print("  - all             : Train all models")
        return

    model_name = sys.argv[1].lower()

    # Individual model training
    if model_name == 'rf':
        results = train_rf()
        plot_results(results)

    elif model_name == 'svm':
        results = train_svm()
        plot_results(results)

    elif model_name == 'lr':
        results = train_lr()
        plot_results(results)

    elif model_name == 'knn':
        results = train_knn()
        plot_results(results)

    elif model_name == 'gb':
        results = train_gb()
        plot_results(results)

    elif model_name == 'xgb':
        results = train_xgb()
        plot_results(results)

    elif model_name == 'mlp':
        results = train_mlp()
        plot_results(results)

    # Train all models
    elif model_name == 'all':
        print("Training all models...")
        all_results = [
            train_rf(),
            train_svm(),
            train_lr(),
            train_knn(),
            train_gb(),
            train_xgb(),
            train_mlp()
        ]
        plot_comparison(all_results)

    else:
        print(f"Unknown model: {model_name}")
        print("Use 'python main.py --help' for available options")


if __name__ == '__main__':
    main()
