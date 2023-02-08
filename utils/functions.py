import numpy as np
import pandas as pd

import keras_tuner as kt
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import os
import shutil
from datetime import datetime
from pathlib import Path
import joblib

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from matplotlib.colors import to_rgba
import plotly.graph_objects as go


def get_tuner(build_model_fn, tuner_type='bayesian', **kwargs):
    """Producing keras tuner based on given model function and keyword arguments

    Parameters
    ----------
    build_model_fn : function
        Function to produce a model

    tuner_type : str
        Type of the tuner. Implemented: 'bayesian', 'hyperband' and 'sklearn'

    Other Parameters
    ----------------
    **kwargs : dict
        Keyword arguments inserted into tuner definition

    Returns
    -------
    keras_tuner
        A specific keras_tuner based on given tuner_type
    """

    # Clear kt_dir directory
    if os.path.isdir('kt_dir'):
        shutil.rmtree('kt_dir')

    if tuner_type == 'bayesian':
        print('Getting Bayesian keras tuner..')

        default_dict = {'objective': kt.Objective("val_auc", direction='max'),
                        'max_trials': 10,
                        'directory': "kt_dir",
                        'project_name': "project_1"}

        tuner_dict = dict(default_dict, **kwargs)
        tuner = kt.BayesianOptimization(build_model_fn, **tuner_dict)

    elif tuner_type == 'hyperband':
        print('Getting Hyperband keras tuner..')

        default_dict = {'objective': kt.Objective("val_auc", direction='max'),
                        'max_epochs': 10,
                        'hyperband_iterations': 1,
                        'directory': "kt_dir",
                        'project_name': "project_1"}

        tuner_dict = dict(default_dict, **kwargs)
        tuner = kt.Hyperband(build_model_fn, **tuner_dict)

    elif tuner_type == 'sklearn':
        print('Getting sklearn keras tuner..')

        default_dict = {'objective': kt.Objective("score", direction='max'),
                        'max_trials': 10}

        tuner_dict = dict(default_dict, **kwargs)
        tuner = kt.tuners.SklearnTuner(oracle=kt.oracles.BayesianOptimizationOracle(**tuner_dict),
                                       hypermodel=build_model_fn,
                                       directory="kt_dir",
                                       project_name="project_1")

    else:
        print(f'{tuner_type} is not implemented')
        return None

    tuner.search_space_summary()
    return tuner


def train_pred_tuner_models(tuner, build_model_fn, train_x, train_y, test_x, ids, n_models=1, n_folds=1,
                            random_seed=False, save_model=False, eval_x=None, callbacks=None, epochs=25,
                            batch_size=128):
    """Training models based on best tuned models from produced tuner search

    Parameters
    ----------
    tuner : keras_tuner
        Keras tuner

    build_model_fn : function
        Function to produce a model

    train_x : numpy.ndarray
        Train features

    train_y : numpy.ndarray
        Train targets

    test_x : numpy.ndarray
        Test features

    ids : pandas.core.frame.DataFrame
        Indexes of examples

    n_models : int, default 1
        Number of used models from tuner

    n_folds : int, default 1
        Number of folds during training

    random_seed : bool, default False
        True : random_seed will be used. False : seed = 2022 will be used

    save_model : bool, default False
        True : trained model will be saved in default location ("models/")

    eval_x : numpy.ndarray, default None
        Evaluation features

    callbacks : list, default None
        List of all callbacks

    epochs : int, default 25
        Number of epochs for training

    batch_size : int, default 128
        Size of batch

    """

    best_hps = tuner.get_best_hyperparameters(n_models)
    all_preds = []
    all_evals = []
    timestamp = str(datetime.now()).replace(':', '.')

    for fold in range(n_folds):
        if random_seed:
            seed = np.random.randint(1000, size=1)
        else:
            seed = 2022

        h_model = build_model_fn(best_hps[fold % n_models])
        X_train, y_train, X_val, y_val = train_test_split(train_x, train_y, test_size=0.2, random_state=seed)

        with tf.device('/GPU:0'):
            history = h_model.fit(X_train, y_train,
                                  validation_data=(X_val, y_val),
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  callbacks=callbacks,
                                  verbose=2)

        preds = h_model.predict(test_x, batch_size=64).reshape(-1)
        all_preds.append(preds)

        if eval_x is not None:
            evals = h_model.predict(eval_x, batch_size=64)
            all_evals.append(evals)

        if save_model:
            Path(f"models/{build_model_fn.__name__}/{timestamp}").mkdir(parents=True, exist_ok=True)
            h_model.save(
                f"models/{build_model_fn.__name__}/{timestamp}/{build_model_fn.__name__}_{str(fold)}_{str(history.history['val_auc'][-1])[2:6]}.h5")

    print('Computing final predictions..')
    preds = np.zeros(len(all_preds[0]))
    for i in range(n_folds):
        preds += np.asarray(all_preds[i])  # np.array([x[0] for x in all_preds[i]])
    preds /= n_folds

    if eval_x is not None:
        evals = np.zeros(len(all_evals[0]))
        for i in range(n_folds):
            evals += np.array([x[0] for x in all_evals[i]])
        evals /= n_folds

        csv_dict = {}
        csv_dict['eval'] = evals
        csv_dict['test'] = preds

        len_diff = len(csv_dict['test']) - len(csv_dict['eval'])
        csv_dict['eval'] = np.append(csv_dict['eval'], np.repeat(np.nan, len_diff))

        prediction_df = pd.DataFrame.from_dict(csv_dict)
        prediction_df.to_csv(f"predictions/{build_model_fn.__name__}_{timestamp}.csv", index=False)

    sub_ = list(zip(np.asarray(ids).reshape(-1), preds))

    submission_df = pd.DataFrame(sub_, columns=['id', 'target'])
    submission_df = submission_df.sort_values(['id'], ascending=[True])
    submission_df.to_csv(f"submissions/{timestamp}.csv", index=False)


def train_pred_hyperopt_model(hyperparameters, classifiers, train_x, train_y, test_x, ids, n_folds=1, model_fit_parameters=[{}],
                              random_seed=False, save_model=False, eval_x=None, train_val_split=0.2):
    """Training models based on best tuned parameters

    Parameters
    ----------
    hyperparameters : list of dictionaries
        Dictionary of all tuned parameters

    classifiers : list of classifiers
        List of Classifiers for corresponding hyperparameters

    train_x : numpy.ndarray
        Train features

    train_y : numpy.ndarray
        Train targets

    test_x : numpy.ndarray
        Test features

    ids : pandas.core.frame.DataFrame
        Indexes of examples

    n_folds : int, default 1
        Number of folds during training

    model_fit_parameters : list of dictionaries
        Model-specific list of parameters used during training

    random_seed : bool, default False
        True : random_seed will be used. False : seed = 2022 will be used

    save_model : bool, default False
        True : trained model will be saved in default location ("models/")

    eval_x : numpy.ndarray, default None
        Evaluation features

    train_val_split : float, default 0.2
        Split value between train and validation sets

    """

    all_preds = []
    all_evals = []
    timestamp = str(datetime.now()).replace(':', '.')
    n_models = len(hyperparameters)
    for fold in range(n_folds):
        print(f'Fold: {fold+1}/{n_folds}')
        if random_seed:
            seed = np.random.randint(1000, size=1)
        else:
            seed = 2022

        classifier = classifiers[fold % n_models]
        hyperparams = hyperparameters[fold % n_models]
        fit_parameters = model_fit_parameters[fold % n_models]

        h_model = classifier(**hyperparams)

        X_train, y_train, X_val, y_val = train_test_split(train_x, train_y, test_size=train_val_split, random_state=seed)

        h_model.fit(X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    **fit_parameters
                    )

        preds_val = h_model.predict_proba(X_val)[:, 1]
        val_auc = (roc_auc_score(y_val, preds_val))

        preds = h_model.predict_proba(test_x)[:, 1]
        all_preds.append(preds)

        if eval_x is not None:
            evals = h_model.predict_proba(eval_x)[:, 1]
            all_evals.append(evals)

        if save_model:
            Path(f"models/{classifiers[0].__name__}/{timestamp}").mkdir(parents=True, exist_ok=True)
            joblib.dump(h_model, f"models/{classifiers[0].__name__}/{timestamp}/{classifiers[0].__name__}_{str(fold)}_{str(val_auc)[2:6]}.pkl")

    print('Computing final predictions..')
    preds = np.zeros(len(all_preds[0]))
    for i in range(n_folds):
        preds += np.asarray(all_preds[i])
    preds /= n_folds

    if eval_x is not None:
        evals = np.zeros(len(all_evals[0]))
        for i in range(n_folds):
            evals += np.array(all_evals[i])
        evals /= n_folds

        csv_dict = {}
        csv_dict['eval'] = evals
        csv_dict['test'] = preds

        len_diff = len(csv_dict['test']) - len(csv_dict['eval'])
        csv_dict['eval'] = np.append(csv_dict['eval'], np.repeat(np.nan, len_diff))

        prediction_df = pd.DataFrame.from_dict(csv_dict)
        prediction_df.to_csv(f"predictions/{classifiers[0].__name__}_{timestamp}.csv", index=False)

    sub_ = list(zip(np.asarray(ids).reshape(-1), preds))

    submission_df = pd.DataFrame(sub_, columns=['id', 'target'])
    submission_df = submission_df.sort_values(['id'], ascending=[True])
    submission_df.to_csv(f"submissions/{timestamp}.csv", index=False)


def pred_load_models_mean(test_x, ids):
    """Producing submission file based on test set and all models stored in models/_submission

    Parameters
    ----------
    test_x : numpy.ndarray
        Test features

    ids : pandas.core.frame.DataFrame
        Indexes of examples

    """

    directory = 'models/_submission'
    all_preds = []
    timestamp = str(datetime.now()).replace(':', '.')
    n_models = 0

    for filename in os.scandir(directory):
        if filename.is_file():
            model_path = filename.path.replace('\\', '/')
            print(f'Predicting values for {model_path}')

            model = tf.keras.models.load_model(model_path)
            preds = model.predict(test_x, batch_size=64)
            all_preds.append(preds)
            n_models += 1

    preds = np.zeros(len(all_preds[0]))
    for i in range(n_models):
        preds += np.array([x[0] for x in all_preds[i]])

    preds /= n_models
    # preds = np.around(preds, decimals=0)

    sub_ = list(zip(ids, preds))
    submission_df = pd.DataFrame(sub_, columns=['id', 'target'])
    submission_df = submission_df.sort_values(['id'], ascending=[True])
    submission_df.to_csv(f"submissions/{timestamp}.csv", index=False)


def plot_history(history, *, n_epochs=None, plot_lr=False, title=None, bottom=None, top=None):
    """Plot (the last n_epochs epochs of) the training history

    Plots loss and optionally val_loss and lr.

    """
    plt.figure(figsize=(15, 6))
    from_epoch = 0 if n_epochs is None else max(len(history['loss']) - n_epochs, 0)

    # Plot training and validation losses
    plt.plot(np.arange(from_epoch, len(history['loss'])), history['loss'][from_epoch:], label='Training loss')
    try:
        plt.plot(np.arange(from_epoch, len(history['loss'])), history['val_loss'][from_epoch:], label='Validation loss')
        best_epoch = np.argmin(np.array(history['val_loss']))
        best_val_loss = history['val_loss'][best_epoch]
        if best_epoch >= from_epoch:
            plt.scatter([best_epoch], [best_val_loss], c='r', label=f'Best val_loss = {best_val_loss:.5f}')
        if best_epoch > 0:
            almost_epoch = np.argmin(np.array(history['val_loss'])[:best_epoch])
            almost_val_loss = history['val_loss'][almost_epoch]
            if almost_epoch >= from_epoch:
                plt.scatter([almost_epoch], [almost_val_loss], c='orange', label='Second best val_loss')
    except KeyError:
        pass
    if bottom is not None: plt.ylim(bottom=bottom)
    if top is not None: plt.ylim(top=top)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower left')
    if title is not None: plt.title(title)

    # Plot learning rate
    if plot_lr and 'lr' in history:
        ax2 = plt.gca().twinx()
        ax2.plot(np.arange(from_epoch, len(history['lr'])), np.array(history['lr'][from_epoch:]), color='g',
                 label='Learning rate')
        ax2.set_ylabel('Learning rate')
        ax2.legend(loc='upper right')

    plt.show()


def plot_feature_importance(importance, names, model_type, max_features=10):
    """Producing submission file based on test set and all models stored in models/_submission

    Parameters
    ----------
    importance : model feature importance
        Model-specific feature importance list

    names : list of strings
        List of all features

    model_type : str
        Name of the model

    max_features : int, default 10
        Number of plotted features

    """
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
    fi_df = fi_df.head(max_features)

    # Define size of bar plot
    plt.figure(figsize=(15, 15))

    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Add chart labels
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('IMPORTANCE')
    plt.ylabel('FEATURE NAMES')


def reduce_mem_usage(df):
    """ Iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Data Frame for memory reduce
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('-' * 50)
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    print('-' * 50)

    return df


def import_data(file):
    """Create a dataframe and optimize its memory usage

    Parameters
    ----------
    file : str
        The file location

    Returns
    -------
    DataFrame
        A Pandas DataFrame object with optimized memory usage
    """

    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df

def plot_distributions_binary_target(data, cols_no=6, figsize=(20,20), remove_cols=()):
    """Plotting feature distributions with recognition of a binary target

    Parameters
    ----------
    data : pandas.core.frame.DataFrame
        Data Frame for calculating distributions

    cols_no : int, default 6
        Number of figures in one row

    figsize : tuple, default (20,20)
        Size of whole figure

    remove_cols : tuple of strings, default ()
        List of columns that won't be taken for calculating distributions

    """

    figure = plt.figure(figsize=figsize)
    features = [col for col in data.columns if data[col].dtype.name != 'category' and col not in remove_cols]
    for num, feat in enumerate(features):
        tmp_0 = data.loc[data.target == 0][feat]
        tmp_1 = data.loc[data.target == 1][feat]

        plt.subplot(np.ceil(data.shape[-1] / cols_no), cols_no, num+1)
        plt.hist(tmp_0, bins=40, label='0', alpha=0.5, color='red')
        plt.hist(tmp_1, bins=40, label='1', alpha=0.5, color='green')
        plt.legend(loc='upper right')
        plt.title(f'{feat}')
    figure.tight_layout(h_pad=1.0, w_pad=0.8)
    plt.show()

def plot_corr_with_target(data):
    """Plotting feature correlations with target

    Parameters
    ----------
    data : pandas.core.frame.DataFrame
        Data Frame for calculating correlations with target variable

    """

    corr=data.corr().round(2)
    corr=corr.iloc[:-1,-1].sort_values(ascending=False)
    pal=sns.color_palette("RdYlBu",32).as_hex()
    pal=[j for i,j in enumerate(pal) if i not in (14,15)]
    rgb=['rgba'+str(to_rgba(i,0.8)) for i in pal]
    fig=go.Figure()
    fig.add_trace(go.Bar(x=corr.index, y=corr, marker_color=rgb,
                         marker_line=dict(color=pal,width=2),
                         hovertemplate='%{x} correlation with Target = %{y}',
                         showlegend=False, name=''))
    fig.update_layout(yaxis_title='Correlation', xaxis_tickangle=45, width=800)
    fig.show()