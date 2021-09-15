import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


def ProbRegression():
  input = tf.keras.layers.Flatten(input_shape=(28,28))
  output = tf.keras.layers.Dense(10, activation=tfp.bijectors.NormalCDF())
  layers=[input,output]
  model=tf.keras.Sequential(layers)
  model.compile(loss='SparseCategoricalCrossentropy', optimizer='adam',metrics='accuracy')
  return model

Model = ProbRegression()
Model.fit(x_train/255,y_train,epochs=200,validation_data=(x_test/255,y_test))
