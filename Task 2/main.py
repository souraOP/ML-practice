import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib

class AdvertisingPlotter:
    def __init__(self) -> None:
        self.columns = ["TV", "Radio", "Newspaper"]

    def distribution_plot(self, data):
        fig, axes = plt.subplots(1, 3, figsize=(15, 7))
        for ax, column_name in zip(axes, self.columns):
            sns.kdeplot(data=data, x=column_name, ax=ax, label=column_name)
            sns.kdeplot(data=data, x="Sales", ax=ax, label="Sales")
            ax.set_xlabel(f"{column_name} vs Sales")
            ax.set_title(f"Distribution plot {column_name} vs Sales")
            ax.legend()

        plt.tight_layout()
        plt.show()

    def pairplot(self, data):
        sns.pairplot(data = data, x_vars=self.columns, y_vars="Sales", kind="reg")
        plt.show()

    def histogram_plot(self, data):
        data.hist()
        plt.show()

class RegressorModelTrainer:
    def __init__(self):
        self.models = [
            ("Linear Regression", LinearRegression()),
            ("Ridge Regression", Ridge()),
            ("Lasso Regression", Lasso()),
            ("Random Forest", RandomForestRegressor()),
            ("Gradient Boosting", GradientBoostingRegressor()),
            ("ElasticNet", ElasticNet())
        ]

    def training_and_evaluation(self, X_train_scaled, X_test_scaled, y_train, y_test):
        res = []
        for name, model in self.models:
            model.fit(X_train_scaled, y_train)
            # after fitting the model i will get the prediction = y_pred
            y_pred = model.predict(X_test_scaled)

            # then calculate the evalution metrics i.e mse, r2 and cross val score
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            crossValScore = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')

            print(f"Model used: {name}")
            print(f"Mean Squared Error: {mse}")
            print(f"R2 Score: {r2}")
            print(f"Cross Validation for R2 (mean value): {crossValScore.mean()}\n\n")

            # plotting the graphs

            plt.figure(figsize = (12, 6))
            plt.plot(np.arange(len(y_test)), y_test, label = "Actual Trend")
            plt.plot(np.arange(len(y_test)), y_pred, label = "Predicted Trend")
            plt.xlabel("Data Index")
            plt.title(f"{name} - Actual vs Predicted Trend")
            plt.legend();
            plt.show();
            print();
            

def loading_dataset(data_path):
    df = pd.read_csv(data_path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df

def labelEncode(y):
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    return y

def preprocess(X, y):
    scaler = StandardScaler();
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler.fit(X_train, y_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, X_train, y_train, y_test, scaler


def main():
    files_directory = os.listdir(os.getcwd())
    data_directory = [file for file in files_directory if file.endswith('.csv')]
    if data_directory:
        data = loading_dataset(data_directory[0])
        print(data.head())
    else:
        print("No CSV file found...")
        return

    # Some EDA steps 

    # data.info()
    # data.shape
    # data.isnull().sum()  # checking for any null data
    # data.duplicated().sum() # checking duplicate data
    # data.describe()
    # data.head()

    plotter = AdvertisingPlotter()
    plotter.distribution_plot(data)
    plotter.pairplot(data)
    plotter.histogram_plot(data)

    y = data["Sales"]
    X = data.drop(columns="Sales", axis=1)


    # encoding and splitting data into train and test
    y = labelEncode(y)

    X_train_scaled, X_test_scaled, X_train, y_train, y_test, scaler = preprocess(X, y)

    model_trainer = RegressorModelTrainer()
    model_trainer.training_and_evaluation(X_train_scaled, X_test_scaled, y_train, y_test)

    gb = GradientBoostingRegressor()
    gb.fit(X_train, y_train)

    tv = float(input("Enter the TV value: "))
    radio = float(input("Enter the radio value: "))
    news = float(input("Enter the Newspaper value: "))

    new_data = pd.DataFrame({
        "TV": [tv],
        "Radio": [radio],
        "Newspaper": [news]
    })

    new_pred_scaled = scaler.transform(new_data)
    new_pred = gb.predict(new_pred_scaled)
    print(f"Predicted Sales: {abs(new_pred)}")

    joblib.dump(gb, 'gradientBoostingRegression.joblib')

if __name__ == "__main__":
    main()