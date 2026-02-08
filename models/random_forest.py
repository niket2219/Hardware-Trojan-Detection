import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from preprocess_data import prepare_data


def train_rf():
    """Train Random Forest classifier"""
    x_train, x_test, y_train, y_test = prepare_data()
    y_train = y_train.reshape(-1)

    model = RandomForestClassifier(n_estimators=5, max_depth=5, random_state=1)

    start = time.time()
    model.fit(x_train, y_train)
    duration = time.time() - start

    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred, average='macro') * 100
    prec = precision_score(y_test, y_pred, average='macro') * 100
    rec = recall_score(y_test, y_pred, average='macro') * 100

    return {
        'name': 'Random Forest',
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'time': duration
    }
