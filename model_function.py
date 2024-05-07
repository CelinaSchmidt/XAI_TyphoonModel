from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def Load_Model(returnvalue):
    df = pd.read_csv(r'C:\Users\celin\Documents\VSC Documents\Thesis\data\analysis\03_new_model_training\03_new_model_training\new_model_training_dataset.csv')

    df = df[(df[["wind_speed"]] != 0).any(axis=1)]
    df = df.drop(columns=["grid_point_id", "typhoon_year"])

    df = df.fillna(0) 

    features = [
        "wind_speed",
        "track_distance",
        "total_houses",
        "rainfall_max_6h",
        "rainfall_max_24h",
        "rwi",
        "mean_slope",
        "std_slope",
        "mean_tri",
        "std_tri",
        "mean_elev",
        "coast_length",
        "with_coast",
        "urban",
        "rural",
        "water",
        "total_pop",
        "percent_houses_damaged_5years",
    ]

    # Split X and y from dataframe features
    X = df[features]
    y = df["percent_houses_damaged"]

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    bins2 = [0, 0.00009, 1, 10, 50, 101]
    samples_per_bin2, binsP2 = np.histogram(df["percent_houses_damaged"], bins=bins2)
    bin_index2 = np.digitize(df["percent_houses_damaged"], bins=binsP2)
    y_input_strat = bin_index2
    y_input_strat[997] = 5

    unique_values, value_counts = np.unique(y_input_strat, return_counts=True)

    # Split dataset into training set and test set
    y_input_strat = y_input_strat

    # added random state
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, df["percent_houses_damaged"], stratify=y_input_strat, test_size=0.2, random_state = 10
    )

    # normalize the data instead
    scaler = MinMaxScaler().fit(X)
    X_scaled2 = scaler.transform(X)
    X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(
        X_scaled2, df["percent_houses_damaged"], stratify=y_input_strat, test_size=0.2, random_state = 10
    )

    df_red = df[features + ["percent_houses_damaged"]]
    # print(df_red.shape)

    if returnvalue == "df":
        return df_red

    data_train, data_test = train_test_split(df_red, test_size=0.2, random_state = 10)
    # print(data_train.shape)

    if returnvalue == "X_train":
        return X_train
    
    if returnvalue == "X_test":
        return X_test
    
    if returnvalue == "X_train_n":
        return X_train_n
    
    if returnvalue == "X_test_n":
        return X_test_n
    
    if returnvalue == "y_train":
        return y_train
    
    if returnvalue == "y_test":
        return y_test
    
    if returnvalue == "features":
        return features
    




    # XGBoost Reduced Overfitting
    xgb = XGBRegressor(
        base_score=0.5,
        booster="gbtree",
        colsample_bylevel=0.8,
        colsample_bynode=0.8,
        colsample_bytree=0.8,
        gamma=3,
        eta=0.01,
        importance_type="gain",
        learning_rate=0.1,
        max_delta_step=0,
        max_depth=4,
        min_child_weight=1,
        missing=1,
        n_estimators=100,
        early_stopping_rounds=10,
        n_jobs=1,
        nthread=None,
        objective="reg:squarederror",
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        seed=None,
        silent=None,
        subsample=0.8,
        verbosity=1,
        eval_metric=["rmse", "logloss"],
        random_state=0,
    )

    eval_set = [(X_train, y_train)]
    xgb_model = xgb.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    if returnvalue == "xgb_model":
        return xgb_model



