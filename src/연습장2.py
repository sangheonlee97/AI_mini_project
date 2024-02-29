import tensorflow as tf
import tensorflow_hub as hub

hub_url = "https://www.kaggle.com/models/google/movinet/frameworks/TensorFlow2/variations/a5-base-kinetics-600-classification/versions/3"

encoder = hub.KerasLayer(hub_url, trainable=True)

inputs = tf.keras.layers.Input(
    shape=[None, None, None, 3],
    dtype=tf.float32,
    name='image')

# [batch_size, 600]
outputs = encoder(dict(image=inputs))

model = tf.keras.Model(inputs, outputs, name='movinet')

model.summary()