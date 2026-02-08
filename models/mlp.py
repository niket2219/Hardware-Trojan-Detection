import time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from preprocess_data import prepare_data


def train_mlp():
    """Train Multilayer Perceptron classifier"""
    x_train, x_test, y_train, y_test = prepare_data()

    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    model = Sequential()
    model.add(Dense(15, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(75, activation='relu'))
    model.add(Dense(y_test_cat.shape[1], activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    start = time.time()
    model.fit(x_train, y_train_cat, epochs=50,
              batch_size=10, shuffle=False, verbose=0)
    duration = time.time() - start

    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    acc = accuracy_score(y_test, y_pred_classes) * 100
    f1 = f1_score(y_test, y_pred_classes, average='macro') * 100
    prec = precision_score(y_test, y_pred_classes, average='macro') * 100
    rec = recall_score(y_test, y_pred_classes, average='macro') * 100

    return {
        'name': 'Multilayer Perceptron',
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'time': duration
    }
