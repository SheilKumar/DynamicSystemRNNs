import numpy as np
import tensorflow as tf 
import os

class Constructor:
    def __init__(self,Data,neurons_lstm,epochs):
        self.neurons_lstm = neurons_lstm
        self.Data = Data 
        self.epochs=epochs
        self.checkpoint_path = "training_checkpoints/"+str(self.neurons_lstm) + "LSTM_Neurons" + str(self.epochs) + "Epochs.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.cp_callback = tf.keras\
            .callbacks\
                .ModelCheckpoint(filepath=self.checkpoint_path,
                                save_weights=True,
                                verbose=1)
        self.baseline() 

    def baseline(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(
                units=self.neurons_lstm, 
                return_sequences=True,
                stateful=True
            ),
            tf.keras.layers.Dense(3)
        ])
        model.compile(
            loss=tf.losses.MeanSquaredError(),
            optimizer=tf.optimizers.Adam(),
            metrics=['accuracy']
        )
        self.model = model
    
    @tf.function
    def fit_model(self):
        self.model.fit(
            x=self.Data.x_train,
            y=self.Data.y_train,
            epochs=self.epochs,
            callbacks=[self.callbacks]
        )
