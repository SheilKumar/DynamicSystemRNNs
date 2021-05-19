import numpy as np
import tensorflow as tf 
import os

class Constructor:
    def __init__(self,Data,neurons_lstm,epochs,baseline=True):
        self.neurons_lstm = neurons_lstm
        self.Data = Data 
        self.epochs=epochs
        self.checkpoint_path = "training_checkpoints/"+str(self.neurons_lstm) + "LSTM_Neurons" + str(self.epochs) + "Epochs.ckpt"
        self.checkpoint_dir = os.path.join(self.checkpoint_path)
        self.cp_callback = tf.keras\
            .callbacks\
                .ModelCheckpoint(filepath=self.checkpoint_path,
                                save_weights_only=True,
                                verbose=2)
        if baseline:
            self.baseline()
        else:
            self.loadWeights()

    def baseline(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(
                units=self.neurons_lstm,
                batch_input_shape = (self.Data.batch_size,None,3),
                return_sequences=True,
                stateful=True
            ),
            tf.keras.layers.Dense(3)
        ])
        self.model.compile(
            loss=tf.losses.MeanSquaredError(),
            optimizer=tf.optimizers.Adam(),
            metrics=['accuracy']
        )
    
    def fitModel(self):
        self.model.fit(x=self.Data.x_train,
        y=self.Data.y_train,
        batch_size = self.Data.batch_size,
        validation_data=(self.Data.x_test,self.Data.y_test),
        epochs=self.epochs,
        callbacks=[self.cp_callback],
        shuffle=False)

    def loadWeights(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(
                units=self.neurons_lstm,
                batch_input_shape=(1,None,3),
                return_sequences=True,
                stateful=True
            ),
            tf.keras.layers.Dense(3)
        ])
        self.model.compile(
            loss=tf.losses.MeanSquaredError(),
            optimizer=tf.optimizers.Adam(),
            metrics=['accuracy']
        )
        self.model.load_weights(self.checkpoint_path)

    def predict(self):
        self.model(self.Data.prediction_data_listen)
        return self.model.predict(self.Data.prediction_x_data)