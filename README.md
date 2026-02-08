`pip install pandas matplotlib keras scikit-learn numpy more-tertools seaborn xgboost`

## Run

We are using **Gradient Boosting, XGBoost, Logistic Regression, K-Nearest Neighbors, Support-Vectors Machine, Random Forest, and Multilayer Perceptron** to classify the samples into Trojan Free and Trojan Infected circuits.

In order to run the code using the above-mentioned algorithms just enter in console the following commands:

`python main.py rf` - Random Forest

`python main.py svm` - Support Vector Machine

`python main.py lr` - Logistic Regression

`python main.py knn` - K-Nearest Neighbors

`python main.py gb` - Gradient Boosting

`python main.py xgb` - XGBoost

`python main.py mlp` - Multilayer Perceptron

To train all models consecutively and compare results:

`python main.py all`

## PCA Analysis

To compare model performance using PCA features only (20 components):

`python pca_model_comparison.py`

To compare models with all features vs PCA features:

`python pca_vs_nonpca_comparison.py`
