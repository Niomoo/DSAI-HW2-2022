import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import keras
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout,BatchNormalization
from sklearn.preprocessing import MinMaxScaler 

def load_data(filename):
    raw_data_df = pd.read_csv(filename, header=0,  usecols=[0,1,2,3], names=['open', 'high', 'low', 'close'])
    df = pd.DataFrame(raw_data_df)
    return df

class Trader():
    def __init__(self, stock):
        self.stock = 0

    def train(self, X_train, y_train):
        keras.backend.clear_session()
        regressor = Sequential()
        regressor.add(LSTM(units = 100, return_sequences=True, input_shape = (X_train.shape[1], 1)))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units = 100, activation='relu'))
        regressor.add(Dense(units = 1))
        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
        regressor.summary()
        history = regressor.fit(X_train, y_train, epochs = 100, batch_size = 16)
        plt.title('train_loss')
        plt.ylabel('loss')
        plt.xlabel('Epoch')
        plt.plot( history.history["loss"])
        return regressor

    def predict_action(self, pre_price, last_price):
        if self.stock > 1 or self.stock < -1:
            sys.exit("Trading invalid")
        if pre_price > last_price:
            if self.stock == 1:
                action = 0
            elif self.stock == 0:
                action = 1
            elif self.stock == -1:
                action = 1
        elif pre_price < last_price:
            if self.stock == 1:
                action = -1
            elif self.stock == 0:
                action = -1
            elif self.stock == -1:
                action = 0
        else:
            action = 0
        self.stock += action
        return action

if __name__ == "__main__":
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--training", default="training.csv", help="input training data file name")
    parser.add_argument("--testing", default="testing.csv", help="input testing data file name")
    parser.add_argument("--output", default="output.csv", help="output file name")
    args = parser.parse_args()

    # The following part is an example.
    # You can modify it at will.

    training_data = load_data(args.training)
    train_set = training_data['open']
    sc = MinMaxScaler(feature_range = (0, 1))
    train_set = train_set.values.reshape(-1,1)
    training_set_scaled = sc.fit_transform(train_set)
 
    trader = Trader(stock=0)
    X_train = []
    y_train = []
    for i in range(10,len(train_set)):
        X_train.append(training_set_scaled[i-10:i-1, 0]) 
        y_train.append(training_set_scaled[i, 0]) 
    X_train, y_train = np.array(X_train), np.array(y_train) 
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    regressor = trader.train(X_train, y_train)

    testing_data = load_data(args.testing)
    test_set = testing_data['open']

    dataset_total = pd.concat((training_data['open'], testing_data['open']), axis=0)
    inputs = dataset_total[len(dataset_total) - len(test_set) - 10:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)

    X_test = []
    for i in range(10, len(inputs)):
        X_test.append(inputs[i-10:i-1, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)


    with open(args.output, "w") as output_file:
        for row in range(len(testing_data)):
            # We will perform your action as the open price in the next day.
            action = trader.predict_action(predicted_stock_price[row], testing_data['open'][row])
            output_file.write('{}\n'.format(action))
