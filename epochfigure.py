from models import Data 
from constructor import Constructor 
import matplotlib.pyplot as plt
import os

Data = Data([1,1,1],30000,0.01)
Data.getData(150,128,num_predictions=300)

for batch_size in [32,256]:
    # Make directories to hold figures 
    path_title = "plots_title/" + str(batch_size) + "batch_size/"
    path_no_title = "plots_no_title/" + str(batch_size) + "batch_size/"
    if not os.path.exists(path_title):
        os.makedirs(path_title)
    if not os.path.exists(path_no_title):
        os.makedirs(path_no_title)

    for lstm_units in [16,32,128,256]:
        model10 = Constructor(Data, lstm_units, 10, baseline=False,  batch_size_user=32) 
        model10.predict()
        model15 = Constructor(Data, lstm_units, 15, baseline=False,  batch_size_user=32) 
        model15.predict()
        model20 = Constructor(Data, lstm_units, 20, baseline=False,  batch_size_user=32) 
        model20.predict()
        model25 = Constructor(Data, lstm_units, 25, baseline=False,  batch_size_user=32)
        model25.predict()



        ##-----------------------------------------------------------------##
        # Plots with title
        fig,axs = plt.subplots(4,figsize=(9,4), sharex=True)
        fig.suptitle('Network Predictions for different epochs over 300 Time Steps')
        
        #  Set the axes limits for y (slightly above and below ranges for true predicted data)
        #  Set axes limits for x (-15 to number of predictions +15)
        for i,ax in enumerate(axs.flat):
            ax.set_ylim([min(Data.prediction_y_data[:,0])-4\
                        ,max(Data.prediction_y_data[:,0])+4])
            ax.set_xlim([-15,Data.num_predictions+15])

        # Plot 1: 10 Epochs 
        axs[0].plot(Data.prediction_y_data[:,0], 'k-', label = "True Values", linewidth=2)
        axs[0].plot(model10.pred_y_hat[:,0], "r--", label = str(lstm_units)+" LSTM Units "+"10 Epochs")
        axs[0].legend(loc=(0,0),frameon=False,fontsize="x-small")


        # Plot 2: 15 Epochs
        axs[1].plot(Data.prediction_y_data[:,0], 'k-', label = "True Values", linewidth=2)
        axs[1].plot(model15.pred_y_hat[:,0], "r--", label = str(lstm_units)+" LSTM Units "+"15 Epochs")
        axs[1].legend(loc=(0,0),frameon=False,fontsize="x-small")


        # Plot 3: 20 Epochs 
        axs[2].plot(Data.prediction_y_data[:,0], 'k-', label = "True Values", linewidth=2)
        axs[2].plot(model20.pred_y_hat[:,0], "r--", label = str(lstm_units)+" LSTM Units "+"20 Epochs")
        axs[2].legend(loc=(0,0),frameon=False,fontsize="x-small")


        # Plot 4: 25 Epochs 
        axs[3].plot(Data.prediction_y_data[:,0], 'k-', label = "True Values", linewidth=2)
        axs[3].plot(model25.pred_y_hat[:,0], "r--", label = str(lstm_units)+" LSTM Units "+"25 Epochs")
        axs[3].legend(loc=(0,0),frameon=False,fontsize="x-small")


        # Create 1 label for all x-axis labels 
        for ax in axs.flat:
            ax.set(xlabel='Time Steps')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()

        # Fixing the legend to the begining of the z coordinate data           


        #plt.show() # might need to show the plots sometimes 

        #Save plot to location
        fig.savefig(path_title + str(lstm_units)+"LSTM_Units"+"ChangingEpochs.png", bbox_inches='tight',transparent=True, dpi=1200)
        # Completely close plots to prepare next plots 
        plt.clf()
        plt.cla()
        plt.close()
        ##-------------------------------------------------------------------##

        ##-----------------------------------------------------------------##
        # Plots with no title
        fig,axs = plt.subplots(4,figsize=(9,4), sharex=True)
        #fig.suptitle('Network Predictions for different epochs over 300 Time Steps')
        
        #  Set the axes limits for y (slightly above and below ranges for true predicted data)
        #  Set axes limits for x (-15 to number of predictions +15)
        for i,ax in enumerate(axs.flat):
            ax.set_ylim([min(Data.prediction_y_data[:,0])-4\
                        ,max(Data.prediction_y_data[:,0])+4])
            ax.set_xlim([-15,Data.num_predictions+15])

        # Plot 1: 10 Epochs 
        axs[0].plot(Data.prediction_y_data[:,0], 'k-', label = "True Values", linewidth=2)
        axs[0].plot(model10.pred_y_hat[:,0], "r--", label = str(lstm_units)+" LSTM Units "+"10 Epochs")
        axs[0].legend(loc=(0,0),frameon=False,fontsize="x-small")


        # Plot 2: 15 Epochs
        axs[1].plot(Data.prediction_y_data[:,0], 'k-', label = "True Values", linewidth=2)
        axs[1].plot(model15.pred_y_hat[:,0], "r--", label = str(lstm_units)+" LSTM Units "+"15 Epochs")
        axs[1].legend(loc=(0,0),frameon=False,fontsize="x-small")


        # Plot 3: 20 Epochs 
        axs[2].plot(Data.prediction_y_data[:,0], 'k-', label = "True Values", linewidth=2)
        axs[2].plot(model20.pred_y_hat[:,0], "r--", label = str(lstm_units)+" LSTM Units "+"20 Epochs")
        axs[2].legend(loc=(0,0),frameon=False,fontsize="x-small")


        # Plot 4: 25 Epochs 
        axs[3].plot(Data.prediction_y_data[:,0], 'k-', label = "True Values", linewidth=2)
        axs[3].plot(model25.pred_y_hat[:,0], "r--", label = str(lstm_units)+" LSTM Units "+"25 Epochs")
        axs[3].legend(loc=(0,0),frameon=False,fontsize="x-small")


        # Create 1 label for all x-axis labels 
        for ax in axs.flat:
            ax.set(xlabel='Time Steps')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()

        # Fixing the legend to the begining of the z coordinate data           


        #plt.show() # might need to show the plots sometimes 

        #Save plot to location
        fig.savefig(path_no_title + str(lstm_units)+"LSTM_Units"+"ChangingEpochs.png", bbox_inches='tight',transparent=True, dpi=1200)
        # Completely close plots to prepare next plots 
        plt.clf()
        plt.cla()
        plt.close()
        ##-------------------------------------------------------------------##