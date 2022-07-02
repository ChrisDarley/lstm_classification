# lstm_classification

Practicing sequence classification of generated sine waves using pytorch LSTMCell class.

I generate 10 setss of wave generation parameters, and create 10,000 time series where each sample is a signal generated from one these 10 sets of signal generation parameters.  I use the LSTMCell class to build a two layer lstm that classifies which set of generation parameters created a given signal.  The lstm achieves 99.8% validation accuracy and 97.9% test accuracy.

This repo was primarily for me to practice implementing lstms in pytorch, which is why I used LSTMCell rather than LSTM.  The same model could be made using LSTM, but I wanted to practice the process of adding each layer indiviudally.  

This repo could be used to explore the viability of lstms in different situations by changing the signal generation parameters.  For example, how long of a sequence will the lstms accurately predict.  You could graph the number of different parameter sets vs the test accuracy.  You could also change the signal generation parameters to add complexity and test model performance.

Run the wave generation first to save new signal files into the directory, and then run through the train loop.
