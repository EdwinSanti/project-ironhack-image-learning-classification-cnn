
from keras.callbacks import ReduceLROnPlateau



# train.py
def compile_model(model):
    model.compile(
        optimizer="adam", 
        loss="SparseCategoricalCrossentropy",
        metrics=["accuracy"],
    )
    return model
                           #changed from 'sgd' to Adam adapts the learning rate
                           # each parameter individually, allowing the model to 
                           #converge faster and achieve better early accuracy than
                           #standard SGD, especially on datasets like CIFAR-10 with small CNNs.
lr_scheduler = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1,
)
                           # reduce learning rate when a metric has stopped improving
                           #  can help the model slow down and fine tune.
    
def train_model(model, x_train, y_train, batch_size, epochs):
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=[lr_scheduler],
    )
    return history


def evaluate_model(model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return loss, accuracy