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
from app_classes import Csv, MyANN
from keras.models import load_model
from sklearn.metrics import classification_report
from pprint import pprint

def main():
    csv_handler = Csv()
    success, csv_file = csv_handler.get_file()
    if not success:
        print("No file selected, exiting...")
        exit()

    target = csv_handler.get_target()
    if target is None:
        print("No target selected, exiting...")
        exit()

    my_ann = MyANN(csv_file, target)

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

if __name__ == "__main__":
    main()
