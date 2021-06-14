# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
import pandas as pd
from typing import Text, Union
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from .utils import PurgedGroupTimeSeriesSplit
from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...log import get_module_logger


class AutoencoderMlpModel(Model):
    """Auto encoder Mlp Model"""

    def __init__(
            self,
            input_dim=158,
            output_dim=1,
            n_splits=5,
            group_gap=2,
            epochs=100,
            es_patience=5,
            batch_size=4096,
            n_models=2,
            verbose=-1,
            **kwargs
    ):
        # Set logger.
        self.logger = get_module_logger("AutoencoderMlpModel")
        self.logger.info("Auto encoder Mlp Model...")

        self.input_dim=input_dim,
        self.output_dim=output_dim,
        self.n_splits = n_splits
        self.group_gap = group_gap
        self.epochs = epochs
        self.es_patience = es_patience
        self.batch_size = batch_size
        self.verbose = verbose
        self.n_models = n_models
        self.params = {'input_dim': input_dim,
                       'output_dim': output_dim,
                       'hidden_units': [96, 96, 896, 448, 448, 256],
                       'dropout_rates': [0.03527936123679956, 0.038424974585075086, 0.42409238408801436,
                                         0.10431484318345882,
                                         0.49230389137187497, 0.32024444956111164, 0.2716856145683449,
                                         0.4379233941604448],
                       'ls': 0,
                       'lr': 1e-3,
                       }
        self.params.update(kwargs)
        self.ckp_path = "./"
        self.fitted = False

    def create_ae_mlp(self, input_dim, output_dim, hidden_units, dropout_rates, ls=1e-2, lr=1e-3):
        inp = tf.keras.layers.Input(shape=(input_dim,))
        x0 = tf.keras.layers.BatchNormalization()(inp)

        encoder = tf.keras.layers.GaussianNoise(dropout_rates[0])(x0)
        encoder = tf.keras.layers.Dense(hidden_units[0])(encoder)
        encoder = tf.keras.layers.BatchNormalization()(encoder)
        encoder = tf.keras.layers.Activation('swish')(encoder)

        decoder = tf.keras.layers.Dropout(dropout_rates[1])(encoder)
        decoder = tf.keras.layers.Dense(input_dim, name='decoder')(decoder)

        x_ae = tf.keras.layers.Dense(hidden_units[1])(decoder)
        x_ae = tf.keras.layers.BatchNormalization()(x_ae)
        x_ae = tf.keras.layers.Activation('swish')(x_ae)
        x_ae = tf.keras.layers.Dropout(dropout_rates[2])(x_ae)

        out_ae = tf.keras.layers.Dense(output_dim, activation='linear', name='ae_action')(x_ae)

        x = tf.keras.layers.Concatenate()([x0, encoder])
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout_rates[3])(x)

        for i in range(2, len(hidden_units)):
            x = tf.keras.layers.Dense(hidden_units[i])(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('swish')(x)
            x = tf.keras.layers.Dropout(dropout_rates[i + 2])(x)

        out = tf.keras.layers.Dense(output_dim, activation='linear', name='action')(x)

        model = tf.keras.models.Model(inputs=inp, outputs=[decoder, out_ae, out])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss={'decoder': tf.keras.losses.MeanSquaredError(),
                            'ae_action': tf.keras.losses.MeanSquaredError(),
                            'action': tf.keras.losses.MeanSquaredError(),
                            },
                      metrics={'decoder': tf.keras.metrics.MeanAbsoluteError(name='mae'),
                               'ae_action': tf.keras.metrics.MeanAbsoluteError(name='mae'),
                               'action': tf.keras.metrics.MeanAbsoluteError(name='mae'),
                               },
                      )

        return model

    def fit(self, dataset: DatasetH):
        df_train, df_valid = dataset.prepare(
            ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )
        X, y = df_train["feature"].values, df_train["label"].values

        model = self.create_ae_mlp(**self.params)
        group_index = df_train["feature"].index.get_level_values('datetime').astype(str)
        gkf = PurgedGroupTimeSeriesSplit(n_splits=self.n_splits, group_gap=self.group_gap)
        for fold, (tr, te) in enumerate(gkf.split(X, y, group_index)):
            self.logger.info(f"Fitting {fold + 1}/{self.n_splits} fold...")
            es = EarlyStopping(monitor='val_action_loss', min_delta=1e-4, patience=self.es_patience, mode='min',
                               baseline=None, restore_best_weights=True, verbose=self.verbose)
            model.fit(X[tr], [X[tr], y[tr], y[tr]],
                      validation_data=(X[te], [X[te], y[te], y[te]]), epochs=self.epochs,
                      batch_size=self.batch_size, callbacks=[es], verbose=self.verbose)
            model.save(f'{self.ckp_path}_{fold}.hdf5')
        self.fitted = True

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self.fitted == False:
            raise ValueError("model is not fitted yet!")

        df_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        models = [tf.keras.models.load_model(f'{self.ckp_path}_{fold}.hdf5') for fold in range(self.n_splits - self.n_models, self.n_splits)]
        pred = np.concatenate([model.predict(df_test.values)[-1] for model in models], axis=1).mean(axis=1)
        pred = pd.Series(np.squeeze(pred), index=df_test.index)

        return pred
