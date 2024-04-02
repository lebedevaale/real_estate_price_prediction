import time
import shap
import nolds
import optuna
import operator
import numpy as np
import pandas as pd
import sklearn as sk
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
from pathlib import Path
import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from contextlib import contextmanager
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error as mse

# Receiving hyperparams for sample modifications
from configparser import ConfigParser
config = ConfigParser()

#---------------------------------------------------------------------------------------------------------------------------------------

def variables_dynamics(data):

    """
    Function for the plotting of the dynamics for the variables

    Inputs:
    --------------------
    data : pd.DataFrame
        Dataframe with columns for the analysis
    groupby : str
        Column to groupby
    mean_only : bool = False
        Whether to plot only means and not min-max

    Prints:
    --------------------
    Dynamics of the variables
    """

    # Creating grid of subplots
    fig = make_subplots(rows = len(data.columns), cols = 1, subplot_titles = data.columns)

    # Scattering returns
    for i, col in enumerate(data.columns):
        fig.add_trace(go.Scatter(x = data.index, y = data[col], mode = 'lines', name = col), row = i + 1, col = 1)

    # Update layout
    fig.update_layout(
        showlegend = False,
        template = 'plotly_dark',
        font = dict(size = 20),
        height = 300 * len(data.columns),
        width = 1200
    )

    # Show the plot
    fig.show()

#---------------------------------------------------------------------------------------------------------------------------------------

def heatmap(data):

    """
    Function for the plotting of the correlation heatmap

    Inputs:
    --------------------
    data : pd.DataFrame
        Dataframe with columns for the analysis
    
    Prints:
    --------------------
    Correlation heatmap
    """

    # Creating grid of subplots
    fig = make_subplots(rows = 1, cols = 2, subplot_titles = ["Pearson Correlation", "Spearman Correlation"])

    # Add trace for each correlation matrix
    z1 = data.corr(method = 'pearson')
    z2 = data.corr(method = 'spearman')
    z = [z1, z2]
    for i in range(len(z)):
        fig.add_trace(go.Heatmap(z = z[i][::-1],
                                 x = data.columns,
                                 y = data.columns[::-1],
                                 text = z[i][::-1].round(2),
                                 texttemplate = "%{text}",
                                 zmin = -1, zmax = 1), 
                                 row = 1, col = i + 1)

    # Update layout
    fig.update_layout(
        showlegend = False,
        template = 'plotly_dark',
        font = dict(size = 14),
        height = 600,
        width = 1600
    )
    fig.update_annotations(font_size = 30)

    # Show the plot
    fig.show()

#-------------------------------------------------------------------------------------------------------

def stationarity(data):

    '''
    Function for the calculation of stationarity of time series

    Inputs:
    --------------------
    data : pd.DataFrame
        Dataframe with columns for the analysis

    Prints:
    --------------------
    res : pd.DataFrame
        Dataframe with results of the stationarity test
    '''

    res = pd.DataFrame(columns = ['Variable', 'DF statistics', 'DF p-value', 'Lyapunov LE', 'Hurst E'])
    for col in data.columns:
        stat = adfuller(data[col])
        res.loc[len(res)] = [col, stat[0], stat[1], nolds.lyap_r(data[col]), nolds.hurst_rs(data[col])]
    
    return res

#---------------------------------------------------------------------------------------------------------------------------------------

def remove_most_insignificant(X_val, X_test, results):
    """
    Function for the removal of the most insignificant variables from the model

    Parameters
    ----------
    X_val : DataFrame
        Set of X for the validation of the model
    X_test : DataFrame
        Set of X for the testing of the model
    results : model
        Fitted statsmodels model

    Returns
    ----------
    X_val : DataFrame
        Optimized set of X for the validation of the model
    X_test : DataFrame
        Optimized set of X for the testing of the model
    """
    # use operator to find the key which belongs to the maximum value in the dictionary:
    max_p_value = max(results.pvalues.iteritems(), key = operator.itemgetter(1))[0]
    # this is the feature you want to drop:
    X_val.drop(columns = max_p_value, inplace = True)
    X_test.drop(columns = max_p_value, inplace = True)

    return X_val, X_test

#---------------------------------------------------------------------------------------------------------------------------------------

def OLS_optimization(Y_val, X_val, Y_test, X_test, p_value_bord:float = 0.05, log:bool = False):
    """
    Function for the optimization of OLS

    Parameters
    ----------
    Y_val, Y_test : array
        Target variable for the regression
    X_val, X_test : DataFrame
        Set of X for the model
    p_value : float = 0.05
        Maximum acceptable p-value for the coefficient

    Returns
    ----------
    results : model
        Fitted statsmodels model
    val_rmse : float
        RMSE score for the validation
    test_rmse : float
        RMSE score for the test
    Y_val_pred : array
        Prediction for the validation
    Y_test_pred : array
        Prediction for the test
    log : bool = False
        Whether target needs to be raised back to exponent
    """
    
    insignificant_feature = True
    while insignificant_feature:
        model = sm.OLS(Y_val, X_val)
        results = model.fit()
        significant = [p_value < p_value_bord for p_value in results.pvalues]
        if all(significant):
            insignificant_feature = False
        else:
            if X_val.shape[1] == 1:  # if there's only one insignificant variable left
                print('No significant features found')
                results = None
                insignificant_feature = False
            else:
                X_val, X_test = remove_most_insignificant(X_val, X_test,  results)
    print(results.summary())

    Y_val_pred = results.predict(X_val)
    Y_test_pred = results.predict(X_test)
    if log == False:
        val_rmse = mse(Y_val_pred, Y_val, squared = False)
        test_rmse = mse(Y_test_pred, Y_test, squared = False)
    else:
        val_rmse = mse(np.exp(Y_val_pred), np.exp(Y_val), squared = False)
        test_rmse = mse(np.exp(Y_test_pred), np.exp(Y_test), squared = False)
    print('Validation score for the stacked model is: ', round(val_rmse, 3))
    print('Test score for the stacked model is: ', round(test_rmse, 3))

    return results, val_rmse, test_rmse, Y_val_pred, Y_test_pred

#---------------------------------------------------------------------------------------------------------------------------------------

@contextmanager
def timer(logger = None, format_str = '{:.3f}[s]', prefix = None, suffix = None):
    if prefix: format_str = str(prefix) + format_str
    if suffix: format_str = format_str + str(suffix)
    start = time.time()
    yield
    d = time.time() - start
    out_str = format_str.format(d)
    if logger:
        logger.info(out_str)
    else:
        print(out_str)

class TreeModel:
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.trn_data = None
        self.val_data = None
        self.model = None

    def train(self,
              params: dict,
              X_train: pd.DataFrame, y_train: np.ndarray,
              X_val: pd.DataFrame, y_val: np.ndarray,
              location: str,
              train_weight: np.ndarray = None,
              val_weight: np.ndarray = None,
              train_params: dict = {}
              ):
        if self.model_type == "lgb":
            self.trn_data = lgb.Dataset(X_train, label=y_train, weight=train_weight)
            self.val_data = lgb.Dataset(X_val, label=y_val, weight=val_weight)
            self.model = lgb.train(params=params,
                                   train_set=self.trn_data,
                                   valid_sets=[self.trn_data, self.val_data],
                                   **train_params)
            self.save = self.model.save_model(location)
        elif self.model_type == "xgb":
            self.trn_data = xgb.DMatrix(X_train, y_train, weight=train_weight)
            self.val_data = xgb.DMatrix(X_val, y_val, weight=val_weight)
            self.model = xgb.train(params=params,
                                   dtrain=self.trn_data,
                                   evals=[(self.trn_data, "train"), (self.val_data, "val")],
                                   **train_params)
            self.save = self.model.save_model(location)
        elif self.model_type == "cat":
            self.trn_data = cat.Pool(X_train, label=y_train, group_id=[0] * len(X_train))
            self.val_data = cat.Pool(X_val, label=y_val, group_id=[0] * len(X_val))
            self.model = cat.CatBoost(params)
            self.model.fit(
                self.trn_data, eval_set=[self.val_data], use_best_model=True, **train_params)
            self.save = self.model.save_model(location)
        else:
            raise NotImplementedError

    def predict(self, X: pd.DataFrame):
        if self.model_type == "lgb":
            return self.model.predict(
                X, num_iteration=self.model.best_iteration)  # type: ignore
        elif self.model_type == "xgb":
            X_DM = xgb.DMatrix(X)
            return self.model.predict(
                X_DM, ntree_limit=self.model.best_ntree_limit)  # type: ignore
        elif self.model_type == "cat":
            return self.model.predict(X)
        else:
            raise NotImplementedError

    @property
    def feature_names_(self):
        if self.model_type == "lgb":
            return self.model.feature_name()
        elif self.model_type == "xgb":
            return list(self.model.get_score(importance_type="gain").keys())
        elif self.model_type == "cat":
             return self.model.feature_names_
        else:
            raise NotImplementedError

    @property
    def feature_importances_(self):
        if self.model_type == "lgb":
            return self.model.feature_importance(importance_type="gain")
        elif self.model_type == "xgb":
            return list(self.model.get_score(importance_type="gain").values())
        elif self.model_type == "cat":
            return self.model.feature_importances_
        else:
            raise NotImplementedError

#---------------------------------------------------------------------------------------------------------------------------------------

def run_train_and_inference(X_train, X_val, X_test, Y_train, Y_val, Y_test, 
                            use_model, model_params, train_params, location, log:bool = False, shaps:bool = True):

    model = TreeModel(model_type = use_model)
    with timer(prefix = "Model training "):
        model.train(
            params = model_params, X_train = X_train, y_train = Y_train,
            X_val = X_val, y_val = Y_val, train_params = train_params,
            location = location)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    if log == False:
        train_score = mse(Y_train, train_pred, squared = False)
        val_score = mse(Y_val, val_pred, squared = False)
        test_score = mse(Y_test, test_pred, squared = False)
    else:
        train_score = mse(np.exp(Y_train), np.exp(train_pred), squared = False)
        val_score = mse(np.exp(Y_val), np.exp(val_pred), squared = False)
        test_score = mse(np.exp(Y_test), np.exp(test_pred), squared = False)
    print(f'Train score for {use_model} is: ', round(train_score, 2))
    print(f'Validation score for {use_model} is: ', round(val_score, 2))
    print(f'Test score for {use_model} is: ', round(test_score, 2))
    scores = {"train": train_score, "val": val_score, "test": test_score}

    if shaps == True:
        explainer = shap.Explainer(model.predict, X_train)
        shap_values = explainer(X_val)
    else:
        shap_values = None

    return train_pred, val_pred, test_pred, scores, shap_values

#---------------------------------------------------------------------------------------------------------------------------------------

def optuna_and_boosting(lag, random_state, directory:str = '',
                        shaps:bool = True):
    
    def objective_lgb(trial):

        param = {
            "learning_rate": trial.suggest_float("learning_rate", 0.2, 0.5, step = 0.05),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log = True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log = True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "max_depth": trial.suggest_int("max_depth", 4, 7)
        }
        param.update(param_lgb)

        train_param = {
            "num_boost_round": 500,
            "early_stopping_rounds": 100,
            "verbose_eval": False
        }

        dtrain = lgb.Dataset(X_train, label = Y_train)
        dvalid = lgb.Dataset(X_test, label = Y_test)
        gbm = lgb.train(param, dtrain, valid_sets = dvalid, **train_param)
        Y_val_pred = gbm.predict(X_val)
        rmse = mse(Y_val_pred, Y_val, squared = False)

        return rmse

    #---------------------------------------------------------------------------------------------------------------------------------------

    def objective_xgb(trial):

        param = {
            "learning_rate": trial.suggest_float("learning_rate", 0.2, 0.5, step = 0.05),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log = True),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log = True),
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "max_depth": trial.suggest_int("max_depth", 4, 7),
            "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
            "eta": trial.suggest_float("eta", 1e-8, 1.0, log = True),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log = True),
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        }
        param.update(param_xgb)

        train_param = {
            "num_boost_round": 500,
            "early_stopping_rounds": 100,
            "verbose_eval": False
        }

        dtrain = xgb.DMatrix(X_train, label = Y_train)
        dvalid = xgb.DMatrix(X_val, label = Y_val)
        gbm = xgb.train(param, dtrain, evals = [(dtrain, 'train') , (dvalid, 'valid')], **train_param)
        Y_val_pred = gbm.predict(dvalid)
        rmse = mse(Y_val_pred, Y_val, squared = False)

        return rmse

    #---------------------------------------------------------------------------------------------------------------------------------------

    def objective_cat(trial):

        param = {
            "learning_rate": trial.suggest_float("learning_rate", 0.2, 0.5, step = 0.05),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
            "depth": trial.suggest_int("depth", 4, 7),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log = True)
        }
        param.update(param_cat)

        train_param = {
            "early_stopping_rounds": 100, 
            "verbose_eval": False
        }

        gbm = cat.CatBoostRegressor(**param)
        gbm.fit(X_train, Y_train, eval_set = [(X_val, Y_val)], **train_param)
        Y_val_pred = gbm.predict(X_val)
        rmse = mse(Y_val_pred, Y_val, squared = False)

        return rmse

    #---------------------------------------------------------------------------------------------------------------------------------------

    def optuna_study(model):

        objectives = {
            "lgb": objective_lgb,
            "xgb": objective_xgb,
            "cat": objective_cat
        }

        sampler = optuna.samplers.TPESampler(seed = random_state)
        study = optuna.create_study(direction = "minimize", sampler = sampler)
        study.optimize(objectives[model], n_trials = 50, n_jobs = 1, gc_after_trial = True, show_progress_bar = True)
        trial = study.best_trial

        return trial

    config.read(directory + 'config.cfg')
    val_size = float(config.get('params', 'val_size'))
    test_size = float(config.get('params', 'test_size'))
    log = bool(config.get('params', 'log'))

    # Load dataset for modelling
    data = pd.read_parquet(directory + 'Data_for_models/final_full.parquet').dropna(subset = [f'target_{lag}_week_fut'])

    # Split dataset on train, validation and test
    Y = data[f'target_{lag}_week_fut']
    X = data.drop(columns = data.columns[data.columns.str.contains('_week_fut')])
    X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(
        X, Y, test_size = test_size, random_state = random_state)
    X_train, X_val, Y_train, Y_val = sk.model_selection.train_test_split(
        X_train, Y_train, test_size = (1 - test_size) * val_size, random_state = random_state)

    # Fixed params for the models
    param_lgb = {
        "verbosity": -1,
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "random_state": random_state,
        # "device": "gpu",
        # "gpu_platform_id": 0,
        # "gpu_device_id": 0
    }

    param_xgb = {
        "verbosity": 0,
        "objective": "reg:squarederror",
        "booster": "gbtree",
        "random_state": random_state
        # "device": "gpu"
    }

    param_cat = {
        "verbose": False,
        "loss_function": "RMSE",
        "bootstrap_type": "Bayesian",
        "random_seed": random_state,
        # "task_type": "GPU", 
        # "devices": "0:1"
    }

    train_param_lxgb = {
        "num_boost_round": 500,
        "early_stopping_rounds": 100,
        "verbose_eval": 100
    }

    train_param_cat = {
        'early_stopping_rounds': 100,
        'verbose_eval': 100
    }

    dirs = {
        'lgb': directory + f'Models/{lag}/lgb.txt',
        'xgb': directory + f'Models/{lag}/xgb.json',
        'cat': directory + f'Models/{lag}/cat'
    }

    print(f'\n LightGBM, {lag} lag:')
    trial_lgb = optuna_study('lgb')
    trial_lgb.params.update(param_lgb)
    train_pred_lgb, val_pred_lgb, test_pred_lgb, score_lgb, shap_lgb = run_train_and_inference(
        X_train, X_val, X_test, Y_train, Y_val, Y_test, "lgb", trial_lgb.params, train_param_lxgb, 
        dirs['lgb'], log = log, shaps = shaps)
    if shaps == True:
        shap.plots.beeswarm(shap_lgb, show = False, color_bar = False)
        plt.savefig(directory + f'Models/{lag}/lgb.png', bbox_inches = 'tight', dpi = 750)
    
    print(f'\n XGBoost, {lag} lag:')
    trial_xgb = optuna_study('xgb')
    trial_xgb.params.update(param_xgb)
    train_pred_xgb, val_pred_xgb, test_pred_xgb, score_xgb, shap_xgb = run_train_and_inference(
        X_train, X_val, X_test, Y_train, Y_val, Y_test, "xgb", trial_xgb.params, train_param_lxgb, 
        dirs['xgb'], log = log, shaps = shaps)
    if shaps == True:
        shap.plots.beeswarm(shap_xgb, show = False, color_bar = False)
        plt.savefig(directory + f'Models/{lag}/xgb.png', bbox_inches = 'tight', dpi = 750)
    
    print(f'\n CatBoost, {lag} lag:')
    trial_cat = optuna_study('cat')
    trial_cat.params.update(param_cat)
    train_pred_cat, val_pred_cat, test_pred_cat, score_cat, shap_cat = run_train_and_inference(
        X_train, X_val, X_test, Y_train, Y_val, Y_test, "cat", trial_cat.params, train_param_cat, 
        dirs['cat'], log = log, shaps = shaps)
    if shaps == True:
        shap.plots.beeswarm(shap_cat, show = False, color_bar = False)
        plt.savefig(directory + f'Models/{lag}/cat.png', bbox_inches = 'tight', dpi = 750)

    stats = pd.DataFrame()
    stats['models'] = ['lgb', 'xgb', 'cat']
    stats['valid'] = [score_lgb["val"], score_xgb["val"], score_cat["val"]]
    stats['test'] = [score_lgb["test"], score_xgb["test"], score_cat["test"]]

    stack_val = pd.DataFrame()
    stack_val['orig'] = Y_val
    stack_val['lgb'] = val_pred_lgb
    stack_val['xgb'] = val_pred_xgb
    stack_val['cat'] = val_pred_cat

    stack_test = pd.DataFrame()
    stack_test['orig'] = Y_test
    stack_test['lgb'] = test_pred_lgb
    stack_test['xgb'] = test_pred_xgb
    stack_test['cat'] = test_pred_cat

    print(f'\n Stacking, {lag} lag:')
    results, val_rmse, test_rmse, Y_val_pred, Y_test_pred = OLS_optimization(
        stack_val.orig, stack_val.drop('orig', axis = 1).copy(), 
        stack_test.orig, stack_test.drop('orig', axis = 1).copy(), log = log)
    results.save(directory + f'Models/{lag}/stacking.pickle')

    stats.loc[len(stats)] = ['stacked', val_rmse, test_rmse]
    print(f'\n Comparison, {lag} lag:')
    print(stats)

    stack_val['stack'] = Y_val_pred
    stack_test['stack'] = Y_test_pred
    stack_val.to_parquet(directory + f'Predictions/{lag}/gb_val.parquet')
    stack_test.to_parquet(directory + f'Predictions/{lag}/gb_test.parquet')