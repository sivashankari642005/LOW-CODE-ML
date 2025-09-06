import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class LinearRegressionModel:
    def __init__(self, data_path, target_column=None, header='infer'):
        print("Loading data...")
        self.data = pd.read_csv(data_path, header=header)
        print(f"Dataset shape: {self.data.shape}")
        print(f"Columns: {self.data.columns.tolist()}")

        if target_column is None:
            target_column = self.data.columns[-1]
            print(f"Using last column as target: {target_column}")

        self.X = self.data.drop(columns=target_column)
        self.Y = self.data[target_column]
        self.model = LinearRegression()
        print("Data loaded successfully.")

    def split_data(self, test_size=0.2, random_state=42):
        print("Splitting data...")
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=test_size, random_state=random_state
        )
        print("Split done.")

    def train_model(self):
        print("Training model...")
        self.model.fit(self.X_train, self.Y_train)
        print("Model trained.")

    def evaluate_model(self):
        print("Evaluating model...")
        predictions = self.model.predict(self.X_test)
        mse = mean_squared_error(self.Y_test, predictions)
        r2 = r2_score(self.Y_test, predictions)

        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"R^2 Score: {r2:.4f}")

if __name__ == '__main__':
    regressor = LinearRegressionModel(
        data_path='your_data.csv',
        target_column=None,    
        header='infer'          
    )

    regressor.split_data()
    regressor.train_model()
    regressor.evaluate_model()
