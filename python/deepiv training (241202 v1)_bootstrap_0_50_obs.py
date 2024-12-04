import asyncio
import telegram

import keras
import pickle 
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from econml.iv.nnet import DeepIV
from keras.models import load_model
from collections import Counter

from tqdm.notebook import tqdm
tqdm.pandas()

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)

from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import LearningRateScheduler

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.config.list_physical_devices('GPU')

from typing import List

def optimize_floats(df: pd.DataFrame) -> pd.DataFrame:
    floats = df.select_dtypes(include=['float64']).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast='float')
    return df

def optimize_ints(df: pd.DataFrame) -> pd.DataFrame:
    ints = df.select_dtypes(include=['int64']).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast='integer')
    return df

def optimize_objects(df: pd.DataFrame, datetime_features: List[str]) -> pd.DataFrame:
    for col in df.select_dtypes(include=['object']):
        if col not in datetime_features:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if float(num_unique_values) / num_total_values < 0.5:
                df[col] = df[col].astype('category')
        else:
            df[col] = pd.to_datetime(df[col])
    return df

def optimize(df: pd.DataFrame, datetime_features: List[str] = []):
    return optimize_floats(optimize_ints(optimize_objects(df, datetime_features)))


def update_dummy_columns(row):
    row['brand_rename_{}'.format(row["brand_rename"])] = 1
    row['yyyy_{}'.format(row["yyyy"])] = 1
    row['mm_{}'.format(row["mm"])] = 1
    row['category_1_{}'.format(row["category_1"])] = 1
    return row

# 1. Load data
print("Data Loading...")
df_model = optimize(pd.read_feather('../data/DeepIV v2.0.0.ftr'))
df_model = df_model[df_model['itt_hour_ln'].notnull()]
print("Data Loaded")

# 2. Prep data
''' oragnize columns and make dummy columns '''
df_model_org = df_model[['product_id', 
                        'itt_hour_ln', # DV
                        'premium_perc', # IV
                        'category_1', 'yyyy', 'mm', 'brand_rename',  # Dummies
                        'msrp_dollar_ln', 'with_release_date', 'days_since_release_ln', # independent variables
                        'likes_count_cumsum_1k' # instrumental variable
                        ] + [col for col in df_model.columns if "VAE" in col] # product vector
]
df_model_org = optimize(pd.get_dummies(df_model_org, columns=['category_1', 'yyyy' ,'mm', 'brand_rename'],  dtype=np.int8))
print("Data Preped")

# 3. bootstrap (50 samples)
for iter in list(range(1, 51)):
    try:
        print("#"*20, "Boostrap {} Started".format(iter), "#"*20)
        # get bootstrap samples with replacement
        print("Making Bootstrap Dataframe...")
        df_model_bootstrap = df_model_org.sample(len(df_model_org), replace=True, random_state=iter)

        # 4. train model
        y = df_model_bootstrap[['itt_hour_ln']].values
        t = df_model_bootstrap[['premium_perc']].values
        x = df_model_bootstrap.drop(columns=['product_id', 
                                        'itt_hour_ln', # y
                                        'premium_perc', # x
                                        'likes_count_cumsum_1k', # instrumental variable
                                    ]).values
        z = df_model_bootstrap[['likes_count_cumsum_1k']].values

        # set seed as 1004 as in the main model
        print("Defining Model")
        random_seed = 1004
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)
        initializer = tf.keras.initializers.GlorotUniform(seed=random_seed)
        treatment_model = keras.Sequential([
                                            keras.layers.Dense(128, activation='relu', input_shape=(x.shape[1] + 1,), kernel_initializer=initializer),
                                            keras.layers.BatchNormalization(),
                                            keras.layers.Dropout(0.2),
            
                                            keras.layers.Dense(64, activation='relu', kernel_initializer=initializer),
                                            keras.layers.BatchNormalization(),
                                            keras.layers.Dropout(0.2),
                                        
                                            keras.layers.Dense(32, activation='relu', kernel_initializer=initializer),
                                            keras.layers.BatchNormalization(),
                                            keras.layers.Dropout(0.2),
                                        ])

        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)
        initializer = tf.keras.initializers.GlorotUniform(seed=random_seed)
        response_model = keras.Sequential([
                                            keras.layers.Dense(128, activation='relu', input_shape=(x.shape[1] + 1,), kernel_initializer=initializer),
                                            keras.layers.BatchNormalization(),
                                            keras.layers.Dropout(0.2),
                                        
                                            keras.layers.Dense(64, activation='relu', kernel_initializer=initializer),
                                            keras.layers.BatchNormalization(),
                                            keras.layers.Dropout(0.2),
                                        
                                            keras.layers.Dense(32, activation='relu', kernel_initializer=initializer),
                                            keras.layers.BatchNormalization(),
                                            keras.layers.Dropout(0.2),
                                        
                                            keras.layers.Dense(1, activation='relu', kernel_initializer=initializer)
                                        ])

        keras_fit_options_1st = { "epochs": 100,
                            "validation_split": 0.2,
                            "batch_size": 128,
                            'verbose':1, 
                            "callbacks": [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True), 
                                        keras.callbacks.CSVLogger('../model/train_history_241202_v1_bootstrap_{}_1st.csv'.format(iter), separator=",", append=False)]}
        keras_fit_options_2nd = { "epochs": 100,
                            "validation_split": 0.2,
                            "batch_size": 128,
                            'verbose':1, 
                            "callbacks": [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True), 
                                        keras.callbacks.CSVLogger('../model/train_history_241202_v1_bootstrap_{}_2nd.csv'.format(iter), separator=",", append=False)]}

        deepIvEst = DeepIV(n_components = 10, # number of gaussians in our mixture density network
                        m = lambda z, x : treatment_model(keras.layers.concatenate([z, x])), # treatment model
                        h = lambda t, x : response_model(keras.layers.concatenate([t, x])),  # response model
                        n_samples = 1, # number of samples to use to estimate the response
                        use_upper_bound_loss = False, # whether to use an approximation to the true loss
                        n_gradient_samples = 1, # number of samples to use in second estimate of the response
                                                # (to make loss estimate unbiased)
                        optimizer=Adam(learning_rate=0.0000001, clipvalue=1.0), 
                        first_stage_options = keras_fit_options_1st, # options for training treatment model
                        second_stage_options = keras_fit_options_2nd) # options for training response model
        
        print("Training Started.")
        deepIvEst.fit(Y=y, T=t, X=x, Z=z)

        deepIvEst._effect_model.save("../model/DeepIV_effect_model_241202_v1_bootstrap_{}.h5".format(iter))
        print("Model Saved")

        async def finish_training():
            TOKEN = '6975289754:AAGeD0ZeDo13wzPNoRVINYhDFuH6OMUCDoI'
            bot = telegram.Bot(token=TOKEN)
            await bot.send_message(1748164923, "Model Saved, Bootstrap {}/{}".format(iter, 50))
        asyncio.run(finish_training())
    except Exception as e:
        async def something_wrong(error):
            TOKEN = '6975289754:AAGeD0ZeDo13wzPNoRVINYhDFuH6OMUCDoI'
            bot = telegram.Bot(token=TOKEN)
            await bot.send_message(1748164923, "Something is wrong. {} iterations. {}".format(iter, error))
        asyncio.run(something_wrong())
        