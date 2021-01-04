# DynamicSystemRNNs

The aim of this project is to use RNN's (Reccurent Neural Networks) to understand dynamic systems. 

## Network Architecture 

```python
lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(256, return_sequences=True),
    # Shape => [batch, time, features]
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.Dense(units=3)
])
```


* Cell type: LSTM
* RNN Units: 256 
* Dense Layer: 3 Neurons 
* Input Shape: \[X,1,3\] X=12,800 during training 
* Output Shape: \[X,1,3\]

The network is currently given 12,800 individual points to be trained. 

## Training Parameters 
```python
lstm_model.compile(loss=tf.losses.MeanSquaredError(),
              optimizer=tf.optimizers.Adam(),
              metrics=[tf.metrics.MeanAbsoluteError()])
```
```python 
history = lstm_model.fit(x_train,y_train, epochs=30, validation_data=(x_test,y_test))
```

* Loss Function: Mean Squared Error
* Optimizer = Adam 
* Metric = Mean Absolute Error 
* Epochs: 30 

## Total Parameters 

* num_features = 3 (x,y,z)
* LSTM_units = 256 (But can be changed in the future)

* LSTM Parameters = $ 4\cdot(\text{LSTM_units})\cdot(\text{LSTM_units}+\text{num_features}+1))=4(256(256+3+3))=266240 $


## Results 

The `teach_lstm` function is used to bring the LSTM internal state as close as possible to just before the data used for closing the loop. The `close_loop` function uses `teach_lstm` inside it before beginning to predicate states.

```python
def teach_lstm(data,start_index,model,trans=400):
  for i in range(trans):
    model.predict(tf.expand_dims(tf.expand_dims(data[start_index-trans+i],0),0))
  return None
```
```python
def close_loop(data,num_loops,model,start_index,trans=400):
  teach_lstm(data,start_index,model,trans)
  old_state = tf.expand_dims(tf.expand_dims(data[start_index],0),0)
  predicted_states = []
  for i in range(num_loops):
    new_state = model.predict(old_state)+old_state
    old_state = new_state
    predicted_states.append(np.squeeze(np.squeeze(old_state,0),0))
  return np.array(predicted_states)
```

* **Inputs**:
    * `data`: Array containing data from the Lorentz attractor. 
    * `num_loops`: Number of states to be predicted.
    * `model`: Model used to predict data.
    * `start_index`: Index number of `data to begin predicting from. (Reccomended to used indexes greater than 16000 so that the model has not seen the points yet)
    * `trans`: Number of previous states used to adjust the LSTM internal state. 

* **Outputs**:
    * `predicted_states`: An `np.array` of predicted states.

### Accuracy Tests 

Beginning with a starting index of 25000 and predicting 100 consecutive time steps. 

```python
init_state = 25000
loops = 100
test = close_loop(states,loops,lstm_model,init_state,400)
```


![](https://i.imgur.com/2zuy3Fx.png)
![](https://i.imgur.com/qs76LGF.png)
![](https://i.imgur.com/iGJGdHn.png)





## Software Used 

* Python 
    - Tensorflow 
* Google Colab 


## Resources

*   [MIT: Introduction To Deep Learning](http://introtodeeplearning.com/) | [![](http://i.imgur.com/0o48UoR.png)](https://github.com/aamini/introtodeeplearning)
*   [Nonlinear Dynamics and Chaos](https://www.amazon.com/Nonlinear-Dynamics-Chaos-Applications-Nonlinearity/dp/0738204536)
* [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory)
