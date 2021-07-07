from models import Data 
from constructor import Constructor 

Data = Data([1,1,1],30000,0.01)
Data.getData(150,128)

for batch_size in [32,256,1024,3840]:
    for epochs in [10,15,20,25]:
        for neurons_lstm in [16,32,64,128,256]:
            model = Constructor(Data,neurons_lstm,epochs,batch_size_user=batch_size)
            model.fitModel()