import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from training import create_model_train
from data_prepare import get_data

#get data
data_close,valid = get_data(train = False)

inputs = data_close[len(data_close) - len(valid) - 40:].values
inputs = inputs.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(0, 1))
inputs  = scaler.transform(inputs)

#create data for testing
X_test = []
for i in range(40,inputs.shape[0]):
    X_test.append(inputs[i-40:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

#prediction
model = create_model_train()
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)
rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))

# find RMS error
print('RMSE value on validation set:',rms)
valid['Predictions'] = closing_price

#plot prediction
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.xlabel('Date',size=20)
plt.ylabel('Stock Price',size=20)
plt.title('Microsoft stock price predict using LSTM',size=20)
plt.legend(['Model Training Data','Actual Price','Predicted price'])
