# meteoGNN
1) LSTM takes four inputs from command line: 1) File of data, 2) Time in future to forecast (6 = 1 hour), Feature to predict ("TEMP", "HUM", "PRO_X", "PRO_Y", "PRESS"), Number of station
Feature to predict name can change according to what dataframe is used
2) Seq 2 Seq takes same inputs as LSTM.
3) Transformer and GNN take inputs in a different order: 1) File of data, 2) Time in future to forecast (6 = 1 hour), Number of station, Feature to predict. In case some input is missing, the program will later ask to input it.
