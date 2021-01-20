import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
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
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

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


    def fit_model(self, epochs, DATA):
        self.epochs = epochs
        self.model.fit(DATA.x_train,DATA.y_train, epochs=epochs, validation_data=(DATA.x_test,DATA.y_test))
    
    def print_stats(self):
        print(self.epochs)



class States:
    "Classs to create data and unperturbed as well as perturbed states"
    def __init__(self,num_loops,start_index,trans=400):  
        self.num_loops = num_loops
        self.trans = trans 
        self.start_index = start_index

    def teach_lstm(self,DATA):
        for i in range(self.trans):
            model.predict(tf.expand_dims(tf.expand_dims(DATA.datapoints[start_index-self.trans+i],0),0))
        return None

    def crete_unperturbed(self,LSTM,DATA):
        old_state = tf.expand_dims(tf.expand_dims(DATA.datapoints[self.start_index],0),0)
        predicted_states = []
        for i in range(self.num_loops):
            new_state = LSTM.model.predict(old_state)+old_state
            old_state = new_state
            predicted_states.append(np.squeeze(np.squeeze(old_state,0),0))
        self.unperturbed = np.array(predicted_states)
    
    def crete_pertrurbed(self,LSTM,DATA,perturbation = [1e-7,0,0]):
        old_state = tf.expand_dims(tf.expand_dims(DATA.datapoints[self.start_index]+perturbation,0),0)
        predicted_states = []
        for i in range(self.num_loops):
            new_state = LSTM.model.predict(old_state)+old_state
            old_state = new_state
            predicted_states.append(np.squeeze(np.squeeze(old_state,0),0))
        self.perturbed = np.array(predicted_states)


class Lyapunov: 
    "Plot the lyapunov exponent for a set LSTM"
    def __init__(self):
        pass

    def calculate_difference(self,States):
        self.difference = np.linalg.norm(States.unperturbed-States.perturbed, axis=1)
        self.log_difference = np.log(self.difference)

    def plot_exponent(self,States):
        self.calculate_difference(States)
        plt.plot(self.log_difference,'r')
        plt.xlabel('Time')
        plt.ylabel('$\log{d}$')
        plt.title('Log difference plot between the purturbed and unperturbed systems')
        plt.legend()
        plt.show()