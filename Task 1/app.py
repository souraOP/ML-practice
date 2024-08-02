import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import joblib

def loading_dataset(file_path):
    df = pd.read_csv(file_path);
    return df

# we need to preprocess our wine dataset 
# creating the function for it

def preprocessing_the_data(data):

    # we are dropping those rows and values with null values instead of fillna i.e. filling it since the aggregrated data might change
    data = data.dropna();
    data = pd.get_dummies(data, drop_first=True)

    # splitting it into features and target
    # we are only interested with the other columns except the quality 
    X = data.drop('quality', axis=1);
    Y = data['quality'];

    # train and test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=42);

    # scaling the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, Y_train, Y_test, scaler

def train_model(X_train, Y_train):
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, Y_train)
        print(name + ' trained')

    return models

def evaluation(models, X_test, Y_test):   # need to the pass the test variables
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mse = mean_squared_error(Y_test, y_pred)
        mae = mean_absolute_error(Y_test, y_pred)
        r2 = r2_score(Y_test, y_pred)
        accuracy = model.score(X_test, Y_test)
        results[name] = {
            'Mean Squared Error is ': mse,
            "Mean Absolute Error is ": mae,
            "R2 Score is ": r2,
            "Accuracy is ": accuracy
        }
    return results

def hyperparameter_tuning(X_train, Y_train):
    param_grids = {
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.6, 0.8, 1.0]
        }
    }

    tuned_model = {}
    for name in ['Random Forest', 'Gradient Boosting']:
        model = models[name]
        param_grid = param_grids[name]
        grid_search = GridSearchCV(model, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, Y_train)
        tuned_model[name] = grid_search.best_estimator_
        

    return tuned_model


def re_evaluation(tuned_models, X_test, Y_test):
    final_results = {}
    for name, model in tuned_models.items():
        y_pred = model.predict(X_test)
        mse = mean_squared_error(Y_test, y_pred)
        mae = mean_absolute_error(Y_test, y_pred)
        r2 = r2_score(Y_test, y_pred)
        accuracy = model.score(X_test, Y_test)
        final_results[name] = {
            'Mean Squared Error is ': mse,
            "Mean Absolute Error is ": mae,
            "R2 Score is ": r2,
            "Accuracy is ": accuracy
        }

    return final_results


def save_the_model(tuned_models, scaler, file_path = "best_wine_soura.pkl"):
    model_data = {
        'Model': tuned_models,
        'scaler': scaler
    }
    joblib.dump(model_data, file_path)

# this is our file path
file_path = "winequalityN.csv"

# Loading the dataset
data = loading_dataset(file_path)
# data.head()
# print(data.isnull().sum())

# Preprocessing the data
X_train_scaled, X_test_scaled, Y_train, Y_test, scaler = preprocessing_the_data(data);

# training our model
models = train_model(X_train_scaled, Y_train)

# evaluating our model
first_result = evaluation(models, X_test_scaled, Y_test)

print("Results are given: ", first_result)


# tuning the hyperparameters
tuned_models = hyperparameter_tuning(X_train_scaled, Y_train)

# again evaluating
final_results = evaluation(tuned_models, X_test_scaled, Y_test)
print("Optimized Results are : ");
print(final_results)


# save the shit out of it

save_the_model(tuned_models['Gradient Boosting'], scaler)