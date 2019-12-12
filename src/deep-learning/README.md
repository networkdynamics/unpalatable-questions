Our deep learning models involve the following neural networks:
- LSTM
- Bi-LSTM
- Stacked Bi-LSTM
- CNN

By default, these models use pre-trained 300-dimensional GloVe embeddings. Functionality to use ELMo instead by passing ```--elmo```

We also implement a dense hybrid version of these models where the neural network's output is combined with a 'dense' vector created from the hand-crafted feature categories in traditional-machine-learning experiments. Their output is then fed to a FeedForward network followed by a softmax function.

To run, pass the name of the model. Allowed names: lstm | bilstm | stacked_bilstm | cnn | dense_lstm | dense_bilstm | dense_stacked_bilstm | dense_cnn | bert | dense_bert | nli_cnn

#### Example runs:
* To run a Bi-directional LSTM using GloVe embeddings: 
```python main.py --model bilstm```

* To run a CNN with ELMo:
```python main.py --model cnn --elmo```

* To run a LSTM with ELMo, save the predictions as a pickle, and save the model:
```python main.py --model lstm --elmo --save_preds --save_model```

* To run a Dense Hybrid LSTM using GloVe embeddings:
```python main.py --model dense_lstm```

If you want to play around with the number of epochs/learning rate/optimizer etc, edit *train.py*.
