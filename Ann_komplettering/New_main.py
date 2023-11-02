from tkinter import *
from tkinter import filedialog
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.layers import Dense, Dropout, Input
from keras.models import Model, Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras import layers
from new_classes import *
from keras.models import load_model
from sklearn.metrics import classification_report
from pprint import pprint


def get_csv():
    # Args:
    #     Hämtar csv att läsa och lägger in den som
    #     en dataframe.

    # Returns:
    #     En dataframe gjord av den hämtade csv-filen.

    root = Tk()
    root.withdraw()  # hide the root window
    # open a file dialog box to choose a file
    file_path = filedialog.askopenfilename(
        initialdir="/",
        title="Select a File",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
    )
    # print the chosen file path
    print("Selected file:", file_path)
    return file_path


def get_target(file_path):
    # Väljer column att förutsäga/predict
    with open(file_path, "r") as f:
        line = f.readline()
    targets = [line.strip() for line in line.split(",")]
    pprint(targets)
    target = input("Choose a target\n")
    while target not in targets:
        try:
            target = input("Choose a column to predict\n")
        except ValueError:
            print("You didn't choose a column to predict. Please select one.\n")
    return target


csv_file = get_csv()

my_ann = MyANN(csv_file, get_target(csv_file))

reg_or_clas = my_ann.identify_target()

# Splitting & tränar datan
my_ann.preprocess_data()


# Bygger modellen och väljer input-dim baserat på X_train.shape
my_ann.build_model()


# skapar history och trained_model på den tränade och fittad data
my_ann.fit()

# Tar fram y_pred/prediction för plottar
my_ann.pred()


# classification
if reg_or_clas == "categorical":
    print(my_ann.plot_model_accuracy())
    print(my_ann.print_classfication_report())

    # Regression
if reg_or_clas == "continuous":
    print(my_ann.Plot_residual_error())
    print(my_ann.plot_predictions_scatter())


my_ann.My_ann_class_attributes()

