# ML2_pr
CS5339 Prediction 2: Singapore Housing Prices Prediction

This is a quick presentation of the interdependcies of the various scripts:

- The 4 training/test .csv files should be placed in input

- In order to use the Jupyter Notebooks, please go through the process of using LSTM_model (this provides the most comprehensive preprocessed datasets)

- Model_src/Model_LSTM_src usage: 
The .sh scripts show in what order one should execute the python scripts in order to have the corresponding data files generated
The files 'HDB_...'/'Private_...'/'LSTM_...' should be used to inspect the models and tune them
The rest is auxiliary function and class definitions

- The 'special' packages we used are xgboost, lightgbm and GPy, all the rest are standard
