from sklearn.preprocessing import MinMaxScaler
import pandas as pd 
import matplotlib.pyplot as plt
from math import ceil
from config import CONFIG

def get_data(train):
    "Extract data from input csv file and create train and test data"
    "Return values as per input is train or not""
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = pd.read_csv(CONFIG.data_path_csv)


    data['Date'] = pd.to_datetime(data.Date,format='%m/%d/%Y %H:%M:%S')
    data.index = data['Date']
    
    # plot given data     
    plt.figure(figsize=(12,8))
    plt.plot(data['Close'], label='Close Price history',color='g')
    plt.xlabel('Date',size=20)
    plt.ylabel('Stock Price',size=20)
    plt.title('Stock Price of Microsoft over the Years',size=25)

    data_close=data[['Close']]
    dataset = data_close.values
    size = data.shape[0]
    
    #traina nd test split
    train=data_close[:ceil(size*CONFIG.train_test_split)]
    valid=data_close[ceil(size* CONFIG.train_test_split):]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    x_train, y_train = [], []
    for i in range(40,len(train)):
        x_train.append(scaled_data[i-40:i,0])
        y_train.append(scaled_data[i,0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
    
    if train == True:
        return x_train,y_train
    else:
        return data_close,valid
