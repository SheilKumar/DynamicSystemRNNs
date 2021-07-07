from models import Data 
from constructor import Constructor 
import matplotlib.pyplot as plt

Data = Data([1,1,1],30000,0.01)
Data.getData(150,128,num_predictions=300)

for epochs in [10,15,20,25]:
    model16 = Constructor(Data,16,epochs,baseline=False)
    model16.predict()
    model32 = Constructor(Data,32,epochs,baseline=False)
    model32.predict()
    model64 = Constructor(Data,64,epochs,baseline=False)
    model64.predict()
    model128 = Constructor(Data,128,epochs,baseline=False)
    model128.predict()
    model256 = Constructor(Data,256,epochs,baseline=False)
    model256.predict()

    #x-value plots
    plt.figure(figsize=(9,4),dpi=1200)
    plt.plot(Data.prediction_y_data[:,0],'k-', label= "True Values", linewidth=2)
    plt.plot(model16.pred_y_hat[:,0],'r--',label = "16 LSTM Units")
    plt.plot(model32.pred_y_hat[:,0],'b--',label = "32 LSTM Units")
    plt.plot(model64.pred_y_hat[:,0],'g--',label = "64 LSTM Units")
    plt.plot(model128.pred_y_hat[:,0],'m--',label = "128 LSTM Units")
    plt.plot(model256.pred_y_hat[:,0],'--', color="orange", label = "256 LSTM Units")
    plt.xlabel('Time Steps')
    plt.ylabel('Predicted $x$ Value')
    plt.legend()

    
    #y-value plots
    plt.figure(figsize=(9,4),dpi=1200)
    plt.plot(Data.prediction_y_data[:,1],'k-', label= "True Values", linewidth=2)
    plt.plot(model16.pred_y_hat[:,1],'r--',label = "16 LSTM Units")
    plt.plot(model32.pred_y_hat[:,1],'b--',label = "32 LSTM Units")
    plt.plot(model64.pred_y_hat[:,1],'g--',label = "64 LSTM Units")
    plt.plot(model128.pred_y_hat[:,1],'m--',label = "128 LSTM Units")
    plt.plot(model256.pred_y_hat[:,1],'--', color="orange", label = "256 LSTM Units")
    plt.xlabel('Time Steps')
    plt.ylabel('Predicted $y$ Value')
    plt.legend()
    plt.savefig("plots300_no_title/"+"Y_Coordinate"+str(epochs)+"Epochs.png", bbox_inches='tight',transparent=True)
    plt.clf()
    plt.cla()
    plt.close()
    
    #z-value plots
    plt.figure(figsize=(9,4),dpi=1200)
    plt.plot(Data.prediction_y_data[:,2],'k-', label= "True Values", linewidth=2)
    plt.plot(model16.pred_y_hat[:,2],'r--',label = "16 LSTM Units")
    plt.plot(model32.pred_y_hat[:,2],'b--',label = "32 LSTM Units")
    plt.plot(model64.pred_y_hat[:,2],'g--',label = "64 LSTM Units")
    plt.plot(model128.pred_y_hat[:,2],'m--',label = "128 LSTM Units")
    plt.plot(model256.pred_y_hat[:,2],'--', color="orange", label = "256 LSTM Units")
    plt.xlabel('Time Steps')
    plt.ylabel('Predicted $z$ Value')
    plt.legend()
    plt.savefig("plots300_no_title/"+"Z_Coordinate"+str(epochs)+"Epochs.png", bbox_inches='tight',transparent=True)
    plt.clf()
    plt.cla()
    plt.close()
    