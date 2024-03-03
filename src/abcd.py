from keras.models import load_model

model = load_model('../movinet_gazua.h5')

model.predict("")