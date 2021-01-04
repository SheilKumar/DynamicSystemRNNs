import numpy
import tensorflow as tf 
import keras 

class LSTM: 
    "Constructor class for all LSTM"
    def __init__(self,neurons_lstm,neurons_dense,loss_fnc=tf.losses.MeanSquaredError() ,optimizer=tf.optimizers.Adam(),metrics=tf.metrics.MeanAbsoluteError()):
        self.neurons_lstm = neurons_lstm
        self.neurons_dense = neurons_dense
        self.loss_fnc=loss_fnc
        self.optimizer=optimizer
        self.metrics = metrics
    
    def create_model(self):
        model = keras.models.Sequential([
            keras.layers.LSTM(self.neurons_lstm, return_sequences=True),
            keras.layers.Dense(self.neurons_dense)
        ])
        model.compile(loss=self.loss_fnc,
                      optimizer=self.optimizer,
                      metrics=self.metrics)
        return model

