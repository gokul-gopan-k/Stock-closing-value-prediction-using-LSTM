import tensorflow as tf
from config import CONFIG
from data_prepare import get_data

def create_model_train():
    " Create a LSTM model and training"
    
    #get inputs for training
    x_train, y_train = get_data(train = True)
    
    model = tf.keras.models.Sequential([
      tf.keras.layers.LSTM(60, return_sequences=True,input_shape=[x_train.shape[1], 1]),
      tf.keras.layers.LSTM(60),
      tf.keras.layers.Dense(1)
    ])

    model.compile(loss="mean_squared_error",
                  optimizer="adam",
                  )
  
    history = model.fit(x_train,y_train,epochs=CONFIG.EPOCHS,verbose =1,batch_size = CONFIG.BATCH_SIZE)

    return model
