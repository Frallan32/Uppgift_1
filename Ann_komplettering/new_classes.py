import pandas as pd
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import numpy as np
from tensorflow import keras
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Class för min ANN modell
class MyANN:
    # init-funktion som skapar objekt som ska användas i classen
    def __init__(
        self,
        data_set,
        target,
        hidden_layer_sizes=(50, 25, -0.25, 12),
        activation="relu",
        loss="binary_crossentropy",
        optimizer="adam",
        batch_size=32,
        epochs=15,
        monitor="val_loss",
        patience=3,
        mode="auto",
        verbose=1,
        use_multiprocessing=False,
    ):
        self._data = pd.read_csv(data_set)
        self._target = target
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.use_multiprocessing = use_multiprocessing
        self.classes_ = None
        self.loss_type = None
        self.best_loss_ = None
        self.features_ = None
        self.n_layers_ = None
        self.n_outputs_ = None
        self.out_activation_ = None
        self.model_ = None

    def preprocess_data(self):
        # splittar datan i X och y
        X = self._data.drop(columns=[self._target])
        y = self._data[self._target]

        # Splittar datan i träning och test data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=101
        )
        # Skapar en scaler och kör fix och transform
        scaler = MinMaxScaler()

        # Fit and transfrom
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    # Bygger en ANN modell som vi ska köra datan i
    def build_model(self):
        input_dim = self.X_train.shape[1]
        model = Sequential()
        # Skapar fist layer baserat på X_train.shape
        fst_layer = self.hidden_layer_sizes[0]

        if isinstance(fst_layer, int):
            model.add(Dense(fst_layer, input_dim=input_dim, activation=self.activation))
        elif isinstance(fst_layer, float):
            model.add(Dropout(abs(fst_layer), input_dim=input_dim))

        # skapar hidden layers baserat på resterande hidden_layers
        for layer in self.hidden_layer_sizes[1:]:
            if isinstance(layer, int):
                model.add(Dense(layer, activation=self.activation))
            elif isinstance(layer, float):
                model.add(Dropout(abs(layer)))

        # Kör Kör olika layers ut beroende på hur många values man vill förutse.
        # activation är också beroende på om man gör classifier eller regression
        model.add(Dense(self.num_classes, activation=self.activation_out))
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=["accuracy"])
        self.model = model
        return self.model

    # Skapar history från modellen så man kan plotta senare
    def model_loss(self):
        return self.history

    # Kör fit med modellen vi byggt, och skapar även early_stop
    def fit(self):
        early_stop = EarlyStopping(
            monitor=self.monitor,
            mode=self.mode,
            patience=self.patience,
            verbose=self.verbose,
        )
        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            callbacks=[early_stop],
            validation_data=(self.X_test, self.y_test),
            epochs=self.epochs,
            batch_size=self.batch_size,
            use_multiprocessing=self.use_multiprocessing,
        )

    # Sparar modellen.
    def save_original(self):
        self.model.save(
            r"C:\Users\andre\OneDrive\Skrivbord\Inlämning_cnn\model.h5",
            save_format="h5",
        )

    # Laddar in  modellen
    def load_original(self):
        self.model = load_model(
            r"C:\Users\andre\OneDrive\Skrivbord\Inlämning_cnn\model.h5", safe_mode=False
        )

    def identify_target(self):
        # Kollar om datan är categorical eller continuous
        # För att bestämma vilken modell som ska användas

        # Kollar target och kollar hur många unika värden som ska förutses
        column_data_types = self._data[self._target].nunique()

        # Classifier
        if int(column_data_types) <= 10:
            # Categorical data type in all columns
            self.num_classes = int(column_data_types) - 1
            self.loss = "binary_crossentropy"
            self.activation_out = "sigmoid"
            return "categorical"

        # Regression
        elif int(column_data_types) >= 11:
            # Numeric data types (integer or float) in all columns
            self.model_type = "continuous"
            self.loss = "mean_squared_error"
            self.num_classes = 1
            self.activation_out = None
            return "continuous"

        else:
            # Unable to determine data type
            self.model_type = "unknown"
            return "unknown"

    # skapar en prediction vi kan använda för att plotta datan
    def pred(self):
        prediction_p = np.round(self.model.predict(self.X_test))
        self.prediction = (prediction_p > 0.5).astype(int)

    # metod som inte fungerade med testraden
    # def model_predict(self, row):
    #     prediction_p = np.round(self.model.predict(np.array([row])).reshape(1, -1))
    #     return (prediction_p > 0.5).astype(int)

    # skriver ut classification_report för att se olika typer av score
    def print_classfication_report(self):
        print(classification_report(y_true=self.y_test, y_pred=self.prediction))

    # plottar ut loss och val_loss i en graf
    def plot_model_accuracy(self):
        plt.plot(self.history.history["loss"], label="train")
        plt.plot(self.history.history["val_loss"], label="test")
        plt.title("Model loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.show()

    # plottar ut accuracy och val_accuracy i en graf
    def plot_learning_curve(self):
        plt.plot(self.history.history["accuracy"])
        plt.plot(self.history.history["val_accuracy"])
        plt.title("Learning Curve")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="upper left")
        plt.show()

    # plottar ut residual error och skriver ut RSME, MSE och MAE
    def Plot_residual_error(self):
        predictions = self.model.predict(self.X_test)
        y_test = self.y_test
        MSE = mean_squared_error(y_test, predictions)
        RSME = np.sqrt(MSE)
        MAE = mean_absolute_error(y_test, predictions)

        print(f"RSME: {RSME}")
        print(f"MSE: {MSE}")
        print(f"MAE: {MAE}")

        y_test = y_test.values.reshape(-1)
        y_pred = predictions.reshape(-1)

        residuals = y_test - y_pred
        sns.histplot(residuals)
        plt.xlabel("Residuals")
        plt.ylabel("Density")
        plt.title("Distribution of Residual Errors")
        plt.show()

    # Plottar ut scatterplot för real data mot test data
    def plot_predictions_scatter(self):
        predictions = self.model.predict(self.X_test)
        y_test = self.y_test
        plt.scatter(y_test, predictions)
        plt.xlabel("Real Values")
        plt.ylabel("Predicted Values")
        plt.title("Scatter Plot of Real Values vs Predicted Values")
        plt.show()

    # Skriver ut info om metoder i klassen och lite annan info som syns tydligt i metoden.
    def My_ann_class_attributes(self):
        # Listing methods/class labels in Class MyAnn
        classes_ = [
            method
            for method in dir(MyANN)
            if callable(getattr(MyANN, method)) and not method.startswith("__")
        ]

        # Getting last loss of model.history
        loss_ = self.history.history["loss"][-1]

        # Getting the best model.lost
        best_loss = min(self.history.history["loss"])

        # List df columns
        features_ = self._data.columns

        # Number of layers in this model
        n_layers = len(self.model.layers)

        # Number of output layer
        n_outputs = self.model.layers[-1].output_shape[-1]

        # What kinde of output activation
        out_activation = self.model.layers[-1].activation.__name__

        print(f"Class labels : {classes_}")
        print(f"Last model loss: {loss_}")
        print(f"Best Model loss : {best_loss}")
        print(f"Dataframe features : {features_}")
        print(f"Numbers of layers in model : {n_layers}")
        print(f"Numbers of output layer : {n_outputs}")
        print(f"Activation on last layer: {out_activation}")
