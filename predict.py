from tensorflow.python.ops.array_ops import batch_gather
from models import Data 
from constructor import Constructor 
import matplotlib.pyplot as plt

Data = Data([1,1,1],30000,0.01)
Data.getData(150,128,num_predictions=300)

for batch_size in [32,256]:
    # Make directories to hold figures 
    path_title = "plots_title/" + batch_size + "batch_size/"
    path_no_title = "plots_no_title/" + batch_size + "batch_size/"
    if not os.path.exists(path_title):
        os.makedirs(path_title)
    if not os.path.exists(path_no_title):
        os.makedirs(path_no_title)
        
    for epochs in [10,15,20,25]:
        model16 = Constructor(Data,16,epochs,baseline=False,batch_size_user=batch_size)
        model16.predict()
        model32 = Constructor(Data,32,epochs,baseline=False,batch_size_user=batch_size)
        model32.predict()
        model64 = Constructor(Data,64,epochs,baseline=False,batch_size_user=batch_size)
        model64.predict()
        model128 = Constructor(Data,128,epochs,baseline=False,batch_size_user=batch_size)
        model128.predict()
        model256 = Constructor(Data,256,epochs,baseline=False,batch_size_user=batch_size)
        model256.predict()


        ##------------------------------------------------------------------##   
        # Plots with title
        fig,axs = plt.subplots(3,figsize=(9,4), sharex=True)
        fig.suptitle('Network Predictions over 300 Time Steps')

        #  Set the axes limits for y (slightly above and below ranges for true predicted data)
        #  Set axes limits for x (-15 to number of predictions +15)
        for i,ax in enumerate(axs.flat):
            ax.set_ylim([min(Data.prediction_y_data[:,i])-4\
                        ,max(Data.prediction_y_data[:,i])+4])
            ax.set_xlim([-15,Data.num_predictions+15])

        # Subplot 1: Predicted x data
        axs[0]\
        .plot(Data.prediction_y_data[:,0],'k-', label= "True Values", linewidth=2)
        axs[0].set(ylabel='$x$ Value')
        axs[0].plot(model16.pred_y_hat[:,0],'r--',label = "16 LSTM Units")
        axs[0].plot(model32.pred_y_hat[:,0],'b--',label = "32 LSTM Units")
        axs[0].plot(model64.pred_y_hat[:,0],'g--',label = "64 LSTM Units")
        axs[0].plot(model128.pred_y_hat[:,0],'m--',label = "128 LSTM Units")
        axs[0].plot(model256.pred_y_hat[:,0],'y--',label = "256 LSTM Units")

        # Subplot 2: Predicted y data
        axs[1]\
        .plot(Data.prediction_y_data[:,1],'k-',label = "True Values", linewidth=2)
        axs[1].set(ylabel='$y$ Value')
        axs[1].plot(model16.pred_y_hat[:,1],'r--',label = "16 LSTM Units")
        axs[1].plot(model32.pred_y_hat[:,1],'b--',label = "32 LSTM Units")
        axs[1].plot(model64.pred_y_hat[:,1],'g--',label = "64 LSTM Units")
        axs[1].plot(model128.pred_y_hat[:,1],'m--',label = "128 LSTM Units")
        axs[1].plot(model256.pred_y_hat[:,1],'y--',label = "256 LSTM Units")

        #Subplot 3 Predicted z data
        axs[2]\
        .plot(Data.prediction_y_data[:,2],'k-',label = "True Values", linewidth=2)
        axs[2].set(ylabel='$z$ Value')
        axs[2].plot(model16.pred_y_hat[:,2],'r--',label = "16 LSTM Units")
        axs[2].plot(model32.pred_y_hat[:,2],'b--',label = "32 LSTM Units")
        axs[2].plot(model64.pred_y_hat[:,2],'g--',label = "64 LSTM Units")
        axs[2].plot(model128.pred_y_hat[:,2],'m--',label = "128 LSTM Units")
        axs[2].plot(model256.pred_y_hat[:,2],'y--',label = "256 LSTM Units")

        # Create 1 label for all x-axis labels 
        for ax in axs.flat:
            ax.set(xlabel='Time Steps')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()

        # Fixing the legend to the begining of the z coordinate data           
        plt.legend(loc=(0,0), frameon=False, fontsize="x-small")

        #plt.show() # might need to show the plots sometimes 

        #Save plot to location
        fig.savefig(path_title + epochs+"Epochs.png", bbox_inches='tight',transparent=True, dpi=1200)
        # Completely close plots to prepare next plots 
        plt.clf()
        plt.cla()
        plt.close()
        ##-------------------------------------------------------------------##

        ##-------------------------------------------------------------------##
        # Plots without a title 
        fig,axs = plt.subplots(3,figsize=(9,4), sharex=True)
        #fig.suptitle('Network Predictions over 300 Time Steps')

        #  Set the axes limits for y (slightly above and below ranges for true predicted data)
        #  Set axes limits for x (-15 to number of predictions +15)
        for i,ax in enumerate(axs.flat):
            ax.set_ylim([min(Data.prediction_y_data[:,i])-4\
                        ,max(Data.prediction_y_data[:,i])+4])
            ax.set_xlim([-15,Data.num_predictions+15])

        # Subplot 1: Predicted x data
        axs[0]\
        .plot(Data.prediction_y_data[:,0],'k-', label= "True Values", linewidth=2)
        axs[0].set(ylabel='$x$ Value')
        axs[0].plot(model16.pred_y_hat[:,0],'r--',label = "16 LSTM Units")
        axs[0].plot(model32.pred_y_hat[:,0],'b--',label = "32 LSTM Units")
        axs[0].plot(model64.pred_y_hat[:,0],'g--',label = "64 LSTM Units")
        axs[0].plot(model128.pred_y_hat[:,0],'m--',label = "128 LSTM Units")
        axs[0].plot(model256.pred_y_hat[:,0],'y--',label = "256 LSTM Units")

        # Subplot 2: Predicted y data
        axs[1]\
        .plot(Data.prediction_y_data[:,1],'k-',label = "True Values", linewidth=2)
        axs[1].set(ylabel='$y$ Value')
        axs[1].plot(model16.pred_y_hat[:,1],'r--',label = "16 LSTM Units")
        axs[1].plot(model32.pred_y_hat[:,1],'b--',label = "32 LSTM Units")
        axs[1].plot(model64.pred_y_hat[:,1],'g--',label = "64 LSTM Units")
        axs[1].plot(model128.pred_y_hat[:,1],'m--',label = "128 LSTM Units")
        axs[1].plot(model256.pred_y_hat[:,1],'y--',label = "256 LSTM Units")

        #Subplot 3 Predicted z data
        axs[2]\
        .plot(Data.prediction_y_data[:,2],'k-',label = "True Values", linewidth=2)
        axs[2].set(ylabel='$z$ Value')
        axs[2].plot(model16.pred_y_hat[:,2],'r--',label = "16 LSTM Units")
        axs[2].plot(model32.pred_y_hat[:,2],'b--',label = "32 LSTM Units")
        axs[2].plot(model64.pred_y_hat[:,2],'g--',label = "64 LSTM Units")
        axs[2].plot(model128.pred_y_hat[:,2],'m--',label = "128 LSTM Units")
        axs[2].plot(model256.pred_y_hat[:,2],'y--',label = "256 LSTM Units")

        # Create 1 label for all x-axis labels 
        for ax in axs.flat:
            ax.set(xlabel='Time Steps')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()

        # Fixing the legend to the begining of the z coordinate data           
        plt.legend(loc=(0,0), frameon=False, fontsize="x-small")

        #plt.show() # might need to show the plots sometimes 

        #Save plot to location
        fig.savefig(path_no_title + epochs +"Epochs.png", bbox_inches='tight',transparent=True, dpi=1200)
        # Completely close plots to prepare next plots 
        plt.clf()
        plt.cla()
        plt.close()

        ##------------------------------------------------------------------##
