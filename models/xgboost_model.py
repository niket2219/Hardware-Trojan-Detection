import time
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from preprocess_data import prepare_data


def train_xgb():
    """Train XGBoost classifier"""
    x_train, x_test, y_train, y_test = prepare_data()
    y_train = y_train.reshape(-1)

    model = XGBClassifier(
        n_estimators=20, use_label_encoder=False, eval_metric='mlogloss')

    start = time.time()
    model.fit(x_train, y_train)
    duration = time.time() - start

    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred, average='macro') * 100
    prec = precision_score(y_test, y_pred, average='macro') * 100
    rec = recall_score(y_test, y_pred, average='macro') * 100

    return {
        'name': 'XGBoost',
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'time': duration
    }
