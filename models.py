from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error

# preprocessing and model code adapted from https://github.com/rodekruis/GlobalTropicalCycloneModel/blob/main/analysis/04_codes_of_article/05_grid_To_mun_regression_model.ipynb
def Load_Model(returnvalue):

    #warnings.filterwarnings("ignore")
    df = pd.read_csv(r'C:\Users\celin\Documents\VSC Documents\Thesis\data\analysis\03_new_model_training\03_new_model_training\new_model_training_dataset.csv')
    if returnvalue == "df_original":
        return df
    
    # Preprocessing
    # Set any values >100% to 100%
    for i in range(len(df)):
        if df.loc[i, "percent_houses_damaged"] > 100:
            df.at[i, "percent_houses_damaged"] = float(100)

    # Fill NaNs with average estimated value of 'rwi'
    df["rwi"].fillna(df["rwi"].mean(), inplace=True)

    # Remove zeros from wind_speed
    df = df[(df[["wind_speed"]] != 0).any(axis=1)]
    df = df.drop(columns=["grid_point_id", "typhoon_year"])
 
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

    df_red = df[features + ["percent_houses_damaged"]] # non scaled
    df_scaled = pd.DataFrame(X_scaled, columns= features)
    df_scaled["percent_houses_damaged"] = y.reset_index(drop=True)
    # print(df_red.shape)

    if returnvalue == "df":
        return df_scaled

    data_train, data_test = train_test_split(df_red, test_size=0.2, random_state = 10)
    # print(data_train.shape)

    if returnvalue == "X_train":
        return pd.DataFrame(X_train, columns=features)
    
    if returnvalue == "X_test":
        return pd.DataFrame(X_test, columns=features)
    
    if returnvalue == "X_train_n":
        return pd.DataFrame(X_train_n, columns=features)
    
    if returnvalue == "X_test_n":
        return pd.DataFrame(X_test_n, columns=features)
    
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
    
    # XGBoost Reduced Overfitting
    xgb_random = XGBRegressor(
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
        # random_state=0
    )
    xgb_random = xgb_random.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    if returnvalue == "xgb_model_noseed":
        return xgb_random

#####################################################################################################
from model_function import Load_Model
from imblearn.under_sampling import RandomUnderSampler
from xgboost.sklearn import XGBRegressor
from xgboost import XGBClassifier
import pandas as pd

class TwoStageXGB:
    def __init__(self, seed = None):
        self.seed = seed

    def fit(self, X_train, y_train):
        from model_function import Load_Model

        # model 1: full xgb model ########################################################
        self.xgb = Load_Model("xgb_model")
        self.features = Load_Model("features")
        # model 2: classifier model ######################################################
        thres = 10.0

        y_train_bool = y_train >= thres
        y_train_bin = (y_train_bool) * 1

        # Define undersampling strategy
        under = RandomUnderSampler(sampling_strategy=0.1)
        # Fit and apply the transform
        X_train_us, y_train_us = under.fit_resample(X_train, y_train_bin)

        # Use XGBClassifier as a Machine Learning model to fit the data
        xgb_model = XGBClassifier(eval_metric=["error", "logloss"])

        eval_set = [(X_train, y_train_bin)]
        self.xgb_model = xgb_model.fit(
                            X_train_us,
                            y_train_us,
                            eval_set=eval_set,
                            verbose=False,
                            )

        # model 3:highest damage xgb model ##############################################
        
        y_pred_train = self.xgb_model.predict(X_train)
        reduced_df = pd.DataFrame(X_train.copy(), columns=self.features)

        reduced_df["percent_houses_damaged"] = y_train.values
        reduced_df["predicted_value"] = y_pred_train

        fliterd_df = reduced_df[reduced_df.predicted_value == 1]

        # Split X and y from dataframe features
        X_r = fliterd_df[self.features]
        y_r = fliterd_df["percent_houses_damaged"]
        
        xgbR = XGBRegressor(
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
                        verbosity=0,
                        eval_metric=["rmse", "logloss"],
                        random_state=0,
                        )

        eval_set = [(X_r, y_r)]
        self.xgbR_model = xgbR.fit(X_r, y_r, eval_set=eval_set, verbose=False)

    def predict(self, X_test):
        if X_test.shape == (18,) or X_test.shape == (1, 18):
            if self.xgb_model.predict(X_test) == 1:
                return self.xgbR_model.predict(X_test)
            else:
                return self.xgb.predict(X_test)
            
        # Make predictions ########################################################################################

        reduced_test_df = pd.DataFrame(X_test.copy(), columns=self.features)
        reduced_test_df["predicted_value"] = self.xgb_model.predict(X_test)

        # high damaged prediction (df1) / not highly damaged prediction (df2)
        fliterd_test_df1 = reduced_test_df[reduced_test_df.predicted_value == 1]
        fliterd_test_df0 = reduced_test_df[reduced_test_df.predicted_value == 0]

        # Use X0 and X1 for the M1 and MR models' predictions
        X1 = fliterd_test_df1[self.features]
        X0 = fliterd_test_df0[self.features]

        # For the output equal to 1 apply MR to evaluate the performance
        y1_pred = self.xgbR_model.predict(X1) # reduced xgb model for pred = 1
        y1_pred = y1_pred.clip(0, 100) # 
        fliterd_test_df1["predicted_percent_damage"] = y1_pred


        # For the output equal to 0 apply M1 to evaluate the performance
        y0_pred = self.xgb.predict(X0) # full xgb model for pred = 0
        y0_pred = y0_pred.clip(0, 100)
        fliterd_test_df0["predicted_percent_damage"] = y0_pred

        # Join two dataframes together
        join_test_dfs = pd.concat([fliterd_test_df0, fliterd_test_df1])
        join_test_dfs = join_test_dfs.sort_index()

        return join_test_dfs["predicted_percent_damage"].values
    
