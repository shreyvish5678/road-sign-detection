# Import libraries
import matplotlib.pyplot as plt 
import numpy as np 
import pickle 
from sklearn.utils import shuffle 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras.models import load_model 
from tensorflow.keras.utils import plot_model 
# Open the training, validation, and test data sets
with open("./data/train.p", mode='rb') as training_data:
  train = pickle.load(training_data)
with open("./data/valid.p", mode='rb') as validation_data:
  valid = pickle.load(validation_data)
with open("./data/test.p", mode='rb') as testing_data:
  test = pickle.load(testing_data)
# Store the features and the labels
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
# Shuffle the image data set
X_train, y_train = shuffle(X_train, y_train)
# Convert the RGB image data set into grayscale
X_train_grscale = np.sum(X_train/3, axis=3, keepdims=True)
X_test_grscale  = np.sum(X_test/3, axis=3, keepdims=True)
X_valid_grscale  = np.sum(X_valid/3, axis=3, keepdims=True)
# Normalize the data set
X_train_grscale_norm = (X_train_grscale - 128)/128
X_test_grscale_norm = (X_test_grscale - 128)/128
X_valid_grscale_norm = (X_valid_grscale - 128)/128
# Build the neural network architecture
cnn_model = tf.keras.Sequential() # Plain stack of layers
cnn_model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3), 
  strides=(3,3), input_shape = img_shape, activation='relu'))
cnn_model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3), 
  activation='relu'))
cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))
cnn_model.add(tf.keras.layers.Dropout(0.25))
cnn_model.add(tf.keras.layers.Flatten())
cnn_model.add(tf.keras.layers.Dense(128, activation='relu'))
cnn_model.add(tf.keras.layers.Dropout(0.5))
cnn_model.add(tf.keras.layers.Dense(43, activation = 'sigmoid'))
# Compile the model
cnn_model.compile(loss='sparse_categorical_crossentropy', optimizer=(
  keras.optimizers.Adam(
  0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)), metrics =[
  'accuracy'])
# Train the model
history = cnn_model.fit(x=X_train_grscale_norm,
  y=y_train,
  batch_size=128,
  epochs=20,
  verbose=1,
  validation_data = (X_valid_grscale_norm,y_valid))
#Save the model
cnn_model.save("./road_sign.h5")
# Reload the model
model = load_model('./road_sign.h5')
# Show the loss value and metrics for the model on the test data set
score = cnn_model.evaluate(X_test_grscale_norm, y_test,verbose=0)
print('Test Accuracy : {:.4f}'.format(score[1]))
# Get the accuracy statistics of the model on the training and validation data
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(len(accuracy))
# Get the predictions for the test data set
predicted_classes = np.argmax(cnn_model.predict(X_test_grscale_norm), axis=-1)
# Retrieve the indices that we will plot
y_true = y_test
# Plot all possible classes using matplotlib
pictures = 0
i = 0
number_of_pics = 0
while pictures < 43:
  if y_true[i] == pictures:
    number_of_pics += 1
    plt.subplot(5,9,number_of_pics)
    plt.imshow(X_test_grscale_norm[i].squeeze(), 
      cmap='gray', interpolation='none')
    plt.title("Predict {}, Actual {}".format(predicted_classes[i], 
      y_true[i]), fontsize=4)
    pictures += 1
  if i < 12629:
    i += 1
  else:
    i = 0
plt.tight_layout()
plt.savefig('./signs.png')
plt.show()

