# train.py
from keras.callbacks import EarlyStopping


def compile_model(model):
    model.compile(
        optimizer="SGD",
        loss="SparseCategoricalCrossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(model, x_train, y_train, batch_size, epochs):
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
    )
    return history

def evaluate_model(model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return loss, accuracy
