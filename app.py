import streamlit as st
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.metrics import r2_score, make_scorer
import pandas as pd


data = pd.read_csv("housing.csv")

prices = data['MEDV']
features = data.drop('MEDV', axis=1)


def performance_metric(y_true, y_predict):
    return r2_score(y_true, y_predict)


def fit_model(X, y):
    cv_sets = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    regressor = DecisionTreeRegressor(random_state=0)

    params = {'max_depth': range(1, 11)}

    scoring_fnc = make_scorer(performance_metric)

    grid = GridSearchCV(regressor,
                        param_grid=params,
                        scoring=scoring_fnc,
                        cv=cv_sets)

    grid = grid.fit(X, y)

    return grid.best_estimator_


model = fit_model(features, prices)


st.title("Boston Housing Price Predictor")
st.write("Enter home details to predict selling price.")

rm = st.number_input("Number of Rooms (RM)", min_value=1.0, max_value=15.0, value=5.0)
lstat = st.number_input("Neighborhood Poverty Level (%) (LSTAT)", min_value=0.0, max_value=50.0, value=15.0)
ptratio = st.number_input("Student-Teacher Ratio (PTRATIO)", min_value=5.0, max_value=30.0, value=15.0)

if st.button("Predict Price"):
    prediction = model.predict([[rm, lstat, ptratio]])
    st.success(f"Predicted Selling Price: ${prediction[0]:,.2f}")
