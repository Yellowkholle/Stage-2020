import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from matplotlib import pyplot as plt
from random import randint
import scipy.optimize as so
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# Preparing the dataset
# Setup train and test splits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Making a copy before flattening for the next code-segment which displays images
x_train_drawing = x_train

image_size = 784 # 28 x 28
x_train = x_train.reshape(x_train.shape[0], image_size)/255. 
x_test = x_test.reshape(x_test.shape[0], image_size)/255.

# Convert class vectors to binary class matrices
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



#I - Sert à voir qqs images, mais facultative
# ###########################################

# II - Premiers réseaux
# #####################


def FC10(lambda0 = (10**(-2)), epochs = 50, plot_precision = True):
    """
    Le premier réseau de neurones proposé dans l'article : le FC10
    lambda0 : dans l'article prend les valeurs : 1, 10**(-2) ou 10**(-4)
    epochs : correspond au temps d'apprentissage
    plot_precision : sert à montrer l'évolution de la précision au cours du temps d'apprentissage
    """
    
    model = Sequential()
    
    model.add(Dense(units=num_classes, activation='softmax', input_shape=(image_size,) , kernel_regularizer=keras.regularizers.l2(lambda0/10)))
    model.summary()
    
    #Entrainer et évaluer le réseau
    
    model.compile(optimizer= "SGD", loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=128, epochs=epochs, verbose=False, validation_split=.1)
    loss, accuracy  = model.evaluate(x_test, y_test, verbose=False)
    
    if plot_precision:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='best')
        plt.show()
    
    print(f'Test loss: {loss:.3}')
    print(f'Test accuracy: {accuracy:.3}')
    
    #renvoyer le modèle entraîné
    return(model)
    

def FC100_100_10(lambda0 = (10**(-5), 10**(-5), 10**(-6)), epochs = 100, plot_precision = False):
    """
    Le deuxième réseau de neurones proposé dans l'article : le FC100-100-10
    lambda0 : le tuple contenant le facteur de la weight decay pour chaque couche
    epochs : correspond au temps d'apprentissage
    plot_precision : sert à montrer l'évolution de la précision au cours du temps d'apprentissage
    """
    
    lambda1, lambda2, lambda3 = lambda0
    
    # model linéaire
    model = Sequential()
    
    # On ajoute les couches
    
    # Cette couche a 100 neurones (units)
    # La fonction d'activation c'est une sigmoide (activation)
    # on ajoute la régularisation "weight decay" de paramètre lamdda1 (kernel_regularizer)
    model.add(Dense(units=100, activation='sigmoid', input_shape=(image_size,), kernel_regularizer=keras.regularizers.l2(lambda1/100)))
    
    # Idem
    model.add(Dense(units=100, activation='sigmoid', kernel_regularizer=keras.regularizers.l2(lambda2/100)))
    model.add(Dense(units=num_classes, activation='softmax', kernel_regularizer=keras.regularizers.l2(lambda3/10)))
    model.summary()
    
    # Entrainer et évaluer le réseau
    model.compile(optimizer= "SGD", loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=128, epochs=epochs, verbose=False, validation_split=.1)
    loss, accuracy  = model.evaluate(x_test, y_test, verbose=False)
    
    if plot_precision:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='best')
        plt.show()
    
    print(f'Test loss: {loss:.3}')
    print(f'Test accuracy: {accuracy:.3}')
    
    #renvoyer le modèle entraîné
    return(model)


def FC200_200_10(lambda0 = (10**(-5), 10**(-5), 10**(-6)), epochs = 50, plot_precision = True):
    """
    Le troisième réseau de neurones proposé dans l'article : le FC200-200-10
    lambda0 : le tuple contenant le facteur de la weight decay pour chaque couche
    epochs : correspond au temps d'apprentissage
    plot_precision : sert à montrer l'évolution de la précision au cours du temps d'apprentissage
    """
    
    model = Sequential()
    
    # The input layer requires the special input_shape parameter which should match
    # the shape of our training data.
    model.add(Dense(units=200, activation='sigmoid', input_shape=(image_size,), kernel_regularizer=keras.regularizers.l2(10**(-5)/200)))
    model.add(Dense(units=200, activation='sigmoid', kernel_regularizer=keras.regularizers.l2(10**(-5)/200)))
    model.add(Dense(units=num_classes, activation='softmax', kernel_regularizer=keras.regularizers.l2(10**(-6)/10)))
    model.summary()
    
    #Entrainer et évaluer le réseau
    
    model.compile(optimizer= "SGD", loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=128, epochs=epochs, verbose=False, validation_split=.1)
    loss, accuracy  = model.evaluate(x_test, y_test, verbose=False)
    
    if plot_precision:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='best')
        plt.show()
    
    print(f'Test loss: {loss:.3}')
    print(f'Test accuracy: {accuracy:.3}')
    
    return(model)
    
    
# III A la recherche d'adversaires
# #################################


def func(x, c, n, sess, loss, grad):
    ad = x.reshape(1,784)                 #reshape x to pass image into the network
    grad_val = sess.run(grad,feed_dict={intensor: ad})
    loss_val = loss.eval(feed_dict={intensor: ad},session = sess)
    y = np.linalg.norm(ad-x_test[n])*c + loss_val
    gra = (np.array(grad_val, dtype = np.float64)+c*(2*ad[0]-2*x_test[n])).flatten()
    return y[0], gra

plt.close()


# get current tf session and initialize the model to evaluate loss and gradient
sess = tf.compat.v1.Session()
tf.compat.v1.keras.backend.set_session(sess)
model1 = FC100_100_10()
#model1.load_weights(checkpoint_path)
testensor = tf.convert_to_tensor([0])   # crapy crapy shity code, this line tells minimizer what adversarial label we want, 0 means 0, 1 means 1, etc.

#tensor construct
intensor = model1.input
outtensor = model1.output
lo = tf.keras.losses.sparse_categorical_crossentropy(testensor,outtensor)
g = tf.compat.v1.gradients(lo, intensor)

# pick c and n
c = 0.01
n = 1000
bounds = [(0,1)]*784

#noise term to be add on to the original image (this is not a must, actually don't add it)
noise = np.random.uniform(size=(28,28))/10
#actual minimization function , see scipy minimize reference for details
ad = so.minimize(func, x_test[n], method = 'L-BFGS-B', jac = True,args=(c, n, sess, lo, g), bounds = bounds)


# In[367]:

# adversarial sample generation evaluation
print(ad)
asample = ad.x.reshape((1,28,28))
ploc = x_test[n].reshape((1,28,28))
print((asample-ploc).sum())
pic = asample[0]*255
plt.imshow(pic, cmap="gray")
plt.show()
print(y_test[n])
print(type(ad.x))
print(len(ad.x))
print(len(x_test[n]))
print(model1.predict(x_test))
print(model1.predict(np.array([ad.x])))


# plot the model, if error, follow the error to install the essential pacakges
#keras.utils.plot_model1(model, to_file='model.png')


# In[245]:

# when quiting the script, make sure the session is closed, so it does not suck up our your computational power
#sess.close()

# s = creation_adversaire(model1, tf.convert_to_tensor(x_train[0]), 9)
# t = s[0] - x_train[0]
# u = np.reshape(t, (28,28))
# 
# #le changement/le filtre
# ax = plt.subplot(1, 4, 1)
# ax.axis('off')
# plt.imshow(u, cmap='Greys')
# 
# #x0
# ax = plt.subplot(1, 4, 2)
# ax.axis('off')
# plt.imshow(np.reshape(x_train[5], (28,28)), cmap='Greys')
# 
# #image sortie
# ax = plt.subplot(1, 4, 3)
# ax.axis('off')
# plt.imshow(np.reshape(s[0], (28,28)), cmap='Greys')
# 
# #image entrée
# ax = plt.subplot(1, 4, 4)
# ax.axis('off')
# plt.imshow(np.reshape(x_train[0], (28,28)), cmap='Greys')
# 
# #inutile
# # ax = plt.subplot(1, 5, 5)
# # ax.axis('off')
# # ploc = [s[0][i] + x_train[0][i] for i in range(784)]
# # plt.imshow(np.reshape(ploc, (28,28)), cmap='Greys')
# plt.show()
# 
# print(model1.predict_classes(np.array([s[0]])))
# print(model1.predict_proba(np.array([s[0]])))
