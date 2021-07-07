import models

LSTM256 = models.LSTM(256,3)
LSTM256.create_model()
Data = models.Data([1,1,1],20000,0.01)
Data.getData()
LSTM256.fit_model(5,Data)
LSTM256.print_stats()
LSTM256.model.summary()
LSTM256_States = models.States(1100,1000)
LSTM256_States.create_unperturbed(LSTM256,Data)
LSTM256_States.create_pertrurbed(LSTM256,Data)
print(LSTM256_States.unperturbed-LSTM256_States.perturbed)
LSTM256_lyapunov = models.Lyapunov(LSTM256_States)
LSTM256_lyapunov.plot_exponent(LSTM256_States)


LSTM_Layer = LSTM256.model.layers[0]
LSTM_Layer.weights

import matplotlib.pyplot as plt

plt.plot(np.linspace(1,10,20),line)