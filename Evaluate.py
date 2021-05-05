from models import LSTM as LSTM
from models import States as States
from models import Lyapunov as Lyapunov
import tensorflow as tf
import numpy as np

class Evaluate:
    def __init__(self,Data,lstm_neurons,epochs,ax,gradient=0.9056,dense_neruons=3):
        self.lstm = LSTM(lstm_neurons,dense_neruons)
        self.lstm.create_model()
        self.lstm.fit_model(epochs,Data)
        self.lstm.print_stats()
        self.states = States(4000,23000)
        self.states.create_unperturbed(self.lstm,Data)
        self.states.create_perturbed(self.lstm,Data)
        self.Lyapunov = Lyapunov(self.states)
        self.Lyapunov.plot_exponent(ax,gradient)
        
class Predict:
    def __init__(self,Data,lstm_neurons,epochs,dense_neurons=3):
        self.Data = Data
        self.lstm = LSTM(lstm_neurons,dense_neurons,epochs)
        self.lstm.create_model()
        self.lstm.fit_model(Data)
        self.PredStates()
        print(self.unperturbed.shape)
    
        
    def PredStates(self):
        old_state = tf.expand_dims(tf.expand_dims(self.Data.datapoints[20000],0),0)
        predicted_states = []
        for i in range(300):
            new_state = self.lstm.model.predict(old_state)+old_state
            old_state = new_state
            predicted_states.append(np.squeeze(np.squeeze(old_state,0),0))
        self.unperturbed = np.array(predicted_states)