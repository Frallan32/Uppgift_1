import unittest
from unittest.mock import patch
from unittest.mock import Mock
from unittest.mock import patch, mock_open
from main_app import *
from app_classes import MyANN, Csv
import numpy as np
from tkinter import filedialog

def test_get_csv(self, mock_askopenfilename):
    mock_askopenfilename.return_value = (r"C:\Users\andre\OneDrive\Skrivbord\Upgift_1_E-P\CSV_filer\Advertising.csv")
    csv_instance = Csv()  # Create an instance of the Csv class
    result = csv_instance.get_file()  # Call the method on the instance
    self.assertEqual(
        result,
        r"C:\Users\andre\OneDrive\Skrivbord\Upgift_1_E-P\CSV_filer\Advertising.csv",
    )

@patch("builtins.input", return_value="sales")
@patch(
    "builtins.open",
    new_callable=unittest.mock.mock_open,
    read_data="sales\n",
    )
def test_init_raises_error_with_wrong_file_path(self):
    with self.assertRaises(FileNotFoundError):
        MyANN(data_set="wrong/path/to/file.csv", target="sales")


@patch("tkinter.filedialog.askopenfilename")
def test_get_target(self, mock_open, mock_input):
    file_path = (
        r"C:\Users\andre\OneDrive\Skrivbord\Upgift_1_E-P\CSV_filer\Advertising.csv"
    )
    result = Csv.get_target(file_path)
    self.assertEqual(result, "sales")




class TestMyANN(unittest.TestCase):
    def setUp(self):
        # Adjust the path to the CSV as necessary
        self.ann = MyANN(
            data_set=r"C:\Users\andre\OneDrive\Skrivbord\Upgift_1_E-P\CSV_filer\Advertising.csv",
            target="sales",
        )

    def test_preprocess_data(self):
        self.ann.preprocess_data()

        # Assert that the train-test split is correct (80%-20% split for the provided data)
        self.assertEqual(len(self.ann.X_train), 160)
        self.assertEqual(len(self.ann.X_test), 40)

        # Test if scaling is done correctly: Values should be between 0 and 1 after MinMaxScaling
        self.assertTrue(np.all((self.ann.X_train >= 0) & (self.ann.X_train <= 1)))
        self.assertTrue(np.all((self.ann.X_test >= 0) & (self.ann.X_test <= 1)))

    def test_get_csv_cancel(self):
        # File dialog will return None if file selection is canceled
        filedialog.askopenfilename = Mock(return_value=None)
        #mock_askopenfilename.return_value = (r"a/ddg")
        had_no_error, csv_file = Csv().get_file()
        self.assertEqual(had_no_error, False)
        self.assertEqual(csv_file, "File selection cancelled.")

    def test_get_csv_not_csv(self):
        # File dialog will return None if file selection is canceled
        filedialog.askopenfilename = Mock(return_value=r"not_csv")
        result = Csv().get_file()
        self.assertEqual(result, (False, 'Selected file is not a CSV.'))

    def test_get_csv_is_success(self):
        # File dialog will return None if file selection is canceled
        filedialog.askopenfilename = Mock(return_value=r"minecraft.csv")
        result = Csv().get_file()
        self.assertEqual(result, (True, "minecraft.csv"))

    def test_get_target_is_success(self):
        with patch("builtins.open", mock_open(read_data="cat,dog,mouse")) as mock_file:
            with patch('builtins.input', return_value="dog"):
                target = Csv().get_target()
        self.assertEqual(target, "dog")

    def test_get_target_is_fail(self):
        with patch("builtins.open", mock_open(read_data="cat,dog,mouse")):
            with patch('builtins.input', return_value=123):
                target = Csv().get_target()
        self.assertEqual(target, None)


    def test_preprocess_data(self):
        ann = MyANN(data_set = "./CSV_filer/Advertising.csv", target = "sales")
        ann.preprocess_data()
        
        # Check if X_train and X_test have been created
        self.assertTrue(hasattr(ann, 'X_train'))
        self.assertTrue(hasattr(ann, 'X_test'))

        # Check if y_train and y_test have been created
        self.assertTrue(hasattr(ann, 'y_train'))
        self.assertTrue(hasattr(ann, 'y_test'))

        # Check if X_train and X_test have been scaled
        self.assertTrue(all([val >= 0.0 for row in ann.X_train for val in row]) and all([val <= 1.0 for row in ann.X_train for val in row]))
        self.assertTrue(all([val >= 0.0 for row in ann.X_test for val in row]) and all([val <= 1.0 for row in ann.X_test for val in row]))


    def test_prediction(self):
        # Kolla att svaret är samma varje gång
        keras.utils.set_random_seed(73) # random seed for NN
        ann = MyANN(data_set = "./CSV_filer/Advertising.csv", target = "sales")
        ann.identify_target()
        ann.preprocess_data()
        ann.build_model()
        ann.fit()
        ann.pred()
        # printade ut svaret och kollar att svaret blir samma varje gång
        expected = np.array([[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]])
        self.assertTrue(all([x == y for x,y in zip(ann.prediction, expected)]))

if __name__ == "__main__":
    unittest.main()
