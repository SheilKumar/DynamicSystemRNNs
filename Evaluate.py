from models import LSTM as LSTM
from models import States as States
from models import Lyapunov as Lyapunov

class Evaluate:
    def __init__(self,Data,lstm_neurons,dense_neruons,epochs,ax,gradient=0.9056):
        self.lstm = LSTM(lstm_neurons,dense_neruons)
        self.lstm.create_model()
        self.lstm.fit_model(epochs,Data)
        self.lstm.print_stats()
        self.states = States(4000,23000)
        self.states.create_unperturbed(self.lstm,Data)
        self.states.create_perturbed(self.lstm,Data)
        self.Lyapunov = Lyapunov(self.states)
        self.Lyapunov.plot_exponent(ax,gradient)