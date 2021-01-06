import numpy as np
import tensorflow as tf 
import keras 
from scipy.integrate import odeint
from sklearn.model_selection import train_test_split



class Data:
    "Constructor class for training and testing data"

    def __init__(self,state0,num_points,delta_t,rho=28,sigma=10,beta=8.0/3.0,): 
        self.state0 = state0
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.datapoints = odeint(self.f,state0,np.arange(0,delta_t*num_points,delta_t))

    def f(self,state, t):
        x, y, z = state  # Unpack the state vector
        return self.sigma * (y - x), x * (self.rho - z) - y, x * y - self.beta * z  # 

    def getData(self,num_time_steps=16000, time_steps_per_batch=1,start_id=1):
        x_data = np.array([self.datapoints[i+start_id:i+start_id+time_steps_per_batch] for i in range(num_time_steps) ])
        y_data = np.array([self.datapoints[i+start_id+1:i+start_id+time_steps_per_batch+1]-self.datapoints[i+start_id:i+start_id+time_steps_per_batch]for i in range(num_time_steps)])
        x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.2,shuffle=False)
        return x_train,x_test,y_train,y_test

class LSTM: 
    "Constructor class for all LSTM"
    def __init__(self,neurons_lstm,neurons_dense,loss_fnc=tf.losses.MeanSquaredError() ,optimizer=tf.optimizers.Adam(),metrics=tf.metrics.MeanAbsoluteError()):
        self.neurons_lstm = neurons_lstm
        self.neurons_dense = neurons_dense
        self.loss_fnc=loss_fnc
        self.optimizer=optimizer
        self.metrics = metrics
        self.epochs = 0
    
    def create_model(self):
        model = keras.models.Sequential([
            keras.layers.LSTM(self.neurons_lstm, return_sequences=True),
            keras.layers.Dense(self.neurons_dense)
        ])
        model.compile(loss=self.loss_fnc,
                      optimizer=self.optimizer,
                      metrics=self.metrics)
        self.model = model


    def fit_model(self, epochs, Data):
        self.epochs = epochs
        self.model.fit(Data[0],Data[2], epochs=epochs, validation_data=(Data[1],Data[3]))
    
    def print_stats(self):
        print(self.epochs)



class States:
    "Classs to create data and unperturbed as well as perturbed states"
    def __init__(self,model,num_loops,trans=400):   
        pass

    def teach_lstm(data,start_index,model,trans=400):
        for i in range(trans):
            model.predict(tf.expand_dims(tf.expand_dims(data[start_index-trans+i],0),0))
        return None

    def close_loop(data,num_loops,model,start_index,trans=400):
        teach_lstm(data,start_index,model,trans)
        old_state = tf.expand_dims(tf.expand_dims(data[start_index],0),0)
        predicted_states = []
        for i in range(num_loops):
            new_state = model.predict(old_state)+old_state
            old_state = new_state
            predicted_states.append(np.squeeze(np.squeeze(old_state,0),0))
        return np.array(predicted_states)

class Lyapunov: 
    "Plot the lyapunov exponent for a set LSTM"
    def __init__(self,data,model):
        pass