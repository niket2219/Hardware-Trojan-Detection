import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from preprocess_data import prepare_data


def train_svm():
    """Train Support Vector Machine classifier"""
    x_train, x_test, y_train, y_test = prepare_data()
    y_train = y_train.reshape(-1)

    model = SVC(kernel="rbf", C=10, gamma=1)

    start = time.time()
    model.fit(x_train, y_train)
    duration = time.time() - start

    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred, average='macro') * 100
    prec = precision_score(y_test, y_pred, average='macro') * 100
    rec = recall_score(y_test, y_pred, average='macro') * 100

    return {
        'name': 'Support Vector Machine',
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'time': duration
    }
