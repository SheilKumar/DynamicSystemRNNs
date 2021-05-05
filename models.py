import numpy as np
import tensorflow as tf 
import os
import matplotlib.pyplot as plt
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

    def getData(self,num_time_steps=16000, time_steps_per_batch=1,start_id=3000):
        self.x_data = np.array([self.datapoints[i+start_id:i+start_id+time_steps_per_batch] for i in range(num_time_steps)])
        self.y_data = np.array([self.datapoints[i+start_id+1:i+start_id+time_steps_per_batch+1]-self.datapoints[i+start_id:i+start_id+time_steps_per_batch]for i in range(num_time_steps)])
        x_train,x_test,y_train,y_test = train_test_split(self.x_data,self.y_data,test_size=0.2,shuffle=False)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

class LSTM: 
    "Constructor class for all LSTM"
    def __init__(self,neurons_lstm,neurons_dense,epochs,loss_fnc=tf.losses.MeanSquaredError() ,optimizer=tf.optimizers.Adam(),metrics=tf.metrics.MeanAbsoluteError()):
        "Constructor class for all LSTM"
        self.neurons_lstm = neurons_lstm
        self.neurons_dense = neurons_dense
        self.loss_fnc=loss_fnc
        self.optimizer=optimizer
        self.metrics = metrics
        self.epochs = epochs
        self.checkpoint_path = "training_checkpoints/"+str(self.neurons_lstm) + "LSTM_Neurons" + str(self.epochs) + "Epochs.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                           save_weights_only=True,
                                                           verbose=1)
        self.create_model()
    
    def create_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(self.neurons_lstm, return_sequences=True),
            tf.keras.layers.Dense(self.neurons_dense)
        ])
        model.compile(loss=self.loss_fnc,
                      optimizer=self.optimizer,
                      metrics  = ['accuracy'])
        self.model = model


    def fit_model(self, DATA):
        self.model.fit(DATA.x_train,DATA.y_train, 
                       epochs=self.epochs, 
                       validation_data=(DATA.x_test,DATA.y_test),
                       callbacks=[self.cp_callback])
    
    def print_stats(self):
        print(f"Number of LSTM units:    {self.neurons_lstm}")
        print(f"Number of Dense neurons: {self.neurons_dense}")
        print(f"Loss Function:           {str(self.loss_fnc)}")
        print(f"Optimizer:               {str(self.optimizer)}")
        print(f"Metrics:                 {str(self.metrics)}")
        print(f"Number of Epochs:        {self.epochs}")



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

    def create_unperturbed(self,LSTM,DATA):
        old_state = tf.expand_dims(tf.expand_dims(DATA.datapoints[self.start_index],0),0)
        predicted_states = []
        for i in range(self.num_loops):
            new_state = LSTM.model.predict(old_state)+old_state
            old_state = new_state
            predicted_states.append(np.squeeze(np.squeeze(old_state,0),0))
        self.unperturbed = np.array(predicted_states)
    
    def create_perturbed(self,LSTM,DATA,perturbation = [1e-7,0,0]):
        old_state = tf.expand_dims(tf.expand_dims(DATA.datapoints[self.start_index]+perturbation,0),0)
        predicted_states = []
        for i in range(self.num_loops):
            new_state = LSTM.model.predict(old_state)+old_state
            old_state = new_state
            predicted_states.append(np.squeeze(np.squeeze(old_state,0),0))
        self.perturbed = np.array(predicted_states)


class Lyapunov: 
    "Plot the lyapunov exponent for a set LSTM"
    def __init__(self,States):
        self.States = States 
        self.calculate_difference(States)     
        pass

    def calculate_difference(self,States):
        self.difference = np.linalg.norm(States.unperturbed-States.perturbed, axis=1)
        self.log_difference = np.log(self.difference)
        self.intercept = self.log_difference[0]

    def plot_exponent(self,ax,gradient):
        axes = ax
        axes.plot(np.linspace(0,self.States.num_loops/100,self.States.num_loops),self.log_difference,'r')
        axes.plot(gradient*np.linspace(0,20,20)-np.absolute(self.intercept),'b')
        axes.set_xlabel('Time')
        axes.set_ylabel('$\log{d}$')
        axes.set_title('Log difference plot between the purturbed and unperturbed systems')
        axes.legend()
        return axes   

    def set_intercept(self,intercept):
        self.intercept = intercept
