import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import time
from random import shuffle
import scipy.optimize as so




# I - Préparation des données
# ###########################

batch_size= 64

(x_train, y_train), (x_test, y_test)=tf.keras.datasets.mnist.load_data()

# Restructuration des donnée d'entrée pour les réseaux FC (Full connected)
x_train=(x_train.reshape(-1, 784)/255).astype(np.float32)
x_test= (x_test.reshape(-1,  784)/255).astype(np.float32)

# Nombre de classes
num_classes = len(np.unique(y_train))

# Services fournis par l'API tf.data (information sur le Web)
train_ds=tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)




# II - Construction des Réseaux de Neurones
# #########################################

# Construction  d'un réseau à 2 couches cachées de 100 neurones chacune

def FC100_100_10(lambda0 = (10**(-5), 10**(-5), 10**(-6)), couches = (100,100)):
    Nh1, Nh2 = couches
    lambda1, lambda2, lambda3 = lambda0
    model = models.Sequential([
        layers.Dense(units=Nh1, activation='sigmoid',input_shape=(784,), kernel_regularizer=tf.keras.regularizers.l2(lambda1/100)),
        layers.Dense(units=Nh2, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(lambda2/100)),
        layers.Dense(units=10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(lambda3/10))
    ])
    
    print("Structure du modele")
    model.summary()
    
    return(model)


# On fait appls aux fonctions "keras" très pratique
optimizer=tf.keras.optimizers.Adam()
loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
train_loss=tf.keras.metrics.Mean()
train_accuracy=tf.keras.metrics.SparseCategoricalAccuracy()
test_loss=tf.keras.metrics.Mean()
test_accuracy=tf.keras.metrics.SparseCategoricalAccuracy()

# ici "@tf.function" perm d'exécute + rapidement les instructions
@tf.function
def train_step(images, labels, model):
  with tf.GradientTape() as tape:
    predictions=model(images)
    loss=loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  train_loss(loss) # + bas dans "def train" on comprend l'intérêt de cette instruction (édition des résultats
  train_accuracy(labels, predictions) # idem

def train(train_ds, nbr_entrainement, model):
  for entrainement in range(nbr_entrainement):
    start=time.time()
    for images, labels in train_ds:
      train_step(images, labels, model)
    message='Entrainement {:04d}, loss: {:6.4f}, accuracy: {:7.4f}%, temps: {:7.4f}'
    print(message.format(entrainement+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         time.time()-start))
    train_loss.reset_states()
    train_accuracy.reset_states()

def test(xt,yt,model):
  start=time.time()
  predictions=model(xt)
  t_loss=loss_object(yt, predictions)
  test_loss(t_loss)
  test_accuracy(yt, predictions)
  message='Loss: {:6.4f}, accuracy: {:7.4f}%, temps: {:7.4f}'
  print(message.format(test_loss.result(),
                       test_accuracy.result()*100,
                       time.time()-start))



# ##############################

# Ceci est un bout de code que je n'ai pas relu/testé étant donné que mon test ne correspond pas à ça ! C'est pour ça que je le laisse sous forme d ecommentaire

# '''
# Calcul du gradient du log de la vraisemblance du chiffre cible (proposé par l'adversaire)
# par rapport aux entrées (image) >  vecteur de 784 composantes
# '''
# def gradientX(image, chiffre_cible):
#   with tf.GradientTape() as tape:
#     tape.watch(image)
#     predictions=model(image)
#     output_adversial = tf.math.log(predictions[0,chiffre_cible]) # le log de la composante correspondant au chiffre_cible noté "tk" dans la note
#     # le log pour obtenir l'équivalent de la "loss function"
#     # indice 0 car prediction.shape = (1,10) et 1e premier indice correspondant à la taille du batch (ici 1 car une seule image fournie)
#   gradients = tape.gradient(output_adversial, image)
#   # gradients.shape = (1,784)
#   return gradients
# 
# 
# def creation_Adversaire1(chiffre_a_reconnaitre, m, model):
#     # Sélection de l'image à perturber
#     image_original = x_test[m]
#     imReelle = image_original.reshape(1,784)
# 
#     imAdversaire = tf.constant(imReelle) # on doit faire un "cast" en tenseur de la structure np.array de imReelle
#                                          # pour utiliser les fonctionnalités de TensorFlow (modele + gradient)
#     n=0
#     alpha = 0.01
#     chiffre_reconnu = np.argmax(model(imAdversaire))
#     # Le + simle des algo : "iterative gradient" (on se limite à 100 itérations)
#     while n < 100 and chiffre_reconnu != chiffre_a_reconnaitre:
#         n=n+1
#         delta = alpha * gradientX(imAdversaire,chiffre_a_reconnaitre)
#         imAdversaire = imAdversaire + delta                                     # !!! Pourquoi un  "+" ??????
#         chiffre_reconnu = np.argmax(model(imAdversaire))
#     
#     imAdversaire = imAdversaire.numpy() # On transforme le tf de TensorFlow en np.array de NumPy
#     Delta = imReelle - imAdversaire
#     EQM =  np.mean(Delta*Delta)
#     
#     if n == 100:
#          print("ca marche pas :( ", n)
#          EQM = -1
#     
#     # # On edite le carré de la norme de Frobenius de la perturbation (moyenné par le nombre de pixels)
#     # print("EQM = ",np.mean(Delta * Delta)) # pour ce cas EQM = 0.0015074897
#     
#     return(imAdversaire, imReelle, EQM)
    
    
    
def gradientX2(model, image, chiffre_cible):
  with tf.GradientTape() as tape:
    tape.watch(image)
    predictions=model(image)
    output_adversial = tf.math.log(1-predictions[0,chiffre_cible]) # le log de la composante correspondant au chiffre_cible noté "tk" dans la note
    # le log pour obtenir l'équivalent de la "loss function"
    # indice 0 car prediction.shape = (1,10) et 1e premier indice correspondant à la taille du batch (ici 1 car une seule image fournie)
  gradients = tape.gradient(output_adversial, image)
  # gradients.shape = (1,784)
  return gradients
    
def creation_Adversaire2(m, model):
    # Sélection de l'image à perturber
    image_original = x_test[m]
    imReelle = image_original.reshape(1,784)

    imAdversaire = tf.constant(imReelle) # on doit faire un "cast" en tenseur de la structure np.array de imReelle
                                         # pour utiliser les fonctionnalités de TensorFlow (modele + gradient)
    # Delta = imReelle - imAdversaire
    # EQM =  np.mean(Delta*Delta)
    # print(EQM)
    
    n=0
    alpha = 0.01
    chiffre_reconnu = np.argmax(model(imAdversaire))
    chiffre_original = y_test[m]
    # Le + simle des algo : "iterative gradient" (on se limite à 100 itérations)
    while n < 1000 and chiffre_reconnu == chiffre_original:
        n=n+1
        delta = alpha * gradientX2(model, imAdversaire,chiffre_original)
        imAdversaire = tf.clip_by_value(imAdversaire + delta, clip_value_min=0, clip_value_max=1)
        chiffre_reconnu = np.argmax(model(imAdversaire))
        
    imAdversaire = imAdversaire.numpy() # On transforme le tf de TensorFlow en np.array de NumPy
    Delta = imReelle - imAdversaire
    EQM =  np.mean(Delta*Delta)
    
    if n == 1000:
         #print("ca marche pas :( ")
         EQM = -1
    
    # # On edite le carré de la norme de Frobenius de la perturbation (moyenné par le nombre de pixels)
    # print("EQM = ",np.mean(Delta * Delta)) # pour ce cas EQM = 0.0015074897
    
    #print(EQM)
    return(imAdversaire, imReelle, EQM)
    


    
def gradientX42(model, image, n):
    chiffre_original = y_test[n]
    with tf.GradientTape() as tape:
        tape.watch(image)
        predictions=model(image)
        loss_val = loss_object(chiffre_original, 1-predictions)
        output_adversial = loss_val
    gradients = tape.gradient(output_adversial, image)
    return gradients
    
    
def func(x, model, c, n):
    ad = x.reshape(1,784)
    imAdversaire = tf.constant(ad)
    predictions=model(imAdversaire)
    
    # on calcule la fonction que l'on souahite minimiser
    loss_val = loss_object(y_test[n], 1-predictions)
    loss_val = loss_val.numpy()
    y = np.linalg.norm(ad-x_test[n])*c + loss_val
    
    #on calcule son gradient
    gra = gradientX42(model, imAdversaire, n)
    gra = (gra.numpy()+c*(2*ad[0]-2*x_test[n])).flatten()
    return y, gra

def IntriguingAdversersarialSansButPrecis(n, model1):
    """ on va chercher à créer un adversaire, pour le moment on ne renvoie rien !!!
    
    n = 1000 l'id de l'image que l'on veut trafiquer
    l = le chiffre que l'on souhaiterait trouver
    
    reseau, lambda0, epochs, plot_precision = sont des paramètres pour l'algorithme d'apprentissage
    
    On pourrait ajouter une précision à atteindre !!!
    On pourrait aussir proposer de trouver le premier adversaire que l'on peut, indépendamment de d'un objectif (ce sont des modifs faciles à faire
    """

    #on convertit plein de choses en tenseur pour pouvoir calculer lo et g
    l = y_test[n]


    #il s'agit désormais de trouver un bon c
    c = 2
    bounds = [(0,1)]*784
    
    predict = model1.predict(np.array([x_test[n]]))
    pic = x_test[n]
    asample = pic.reshape((1,28,28))
    ploc = pic.reshape((1,28,28))

    ad = so.minimize(func, x_test[n], method = 'L-BFGS-B', jac = True,args=(model1, 0, n), bounds = bounds)
    
    asample = ad.x.reshape((1,28,28))
    ploc = x_test[n].reshape((1,28,28))
    predict1 = model1.predict(np.array([ad.x]))
    
    possible = (np.argmax(predict1) != l)
    
    
    if not(np.argmax(predict) == l):
        return(0)
    
    elif not(possible):
        return(-1)
    
    while np.argmax(predict) == l:
        #on s'arrête dès que l'on atteint un label différent du bon

        #on minimise la fonction
        ad = so.minimize(func, x_test[n], method = 'L-BFGS-B', jac = True,args=(model1, c, n), bounds = bounds)
        
        asample = ad.x.reshape((1,28,28))
        ploc = x_test[n].reshape((1,28,28))
        predict = model1.predict(np.array([ad.x]))
        c = c/1.5
    
    Delta = asample-ploc
    # print("c = ", c)
    #print(n, "EQM = ",np.mean(Delta * Delta))
    # print("prediction image : ", np.argmax(predict))
    # securite = model1.predict(asample.reshape(1,784))
    # print("prediction : ", predict)
    # print("securite : ", securite)
    # pic = asample[0]*255
    # plt.imshow(pic, cmap="gray")
    # plt.show()
    return(np.mean(Delta * Delta))
    
    
# ###################

nbr_entrainement= 100
model = FC100_100_10(couches = (100,100))
print("Entrainement")
train(train_ds, nbr_entrainement, model)
print("Jeu de test")
test(x_test,y_test, model)
    
minimisons = [x for x in range(len(y_test))]
shuffle(minimisons)
liste1 = []
liste2 = []
for i in minimisons[:1000]:
    retour_adversaire1 = creation_Adversaire2(i, model)
    retour_adversaire2 = IntriguingAdversersarialSansButPrecis(i, model)
    liste1.append(retour_adversaire1[-1])
    liste2.append(retour_adversaire2)

liste1 = [i for i in liste1 if i != 0]
a1 = len(liste1)
print("Il y a : ", 1000-a1, " prédictions fausses au départ, donc inutile de leur chercher un adversaire")
liste2 = [i for i in liste2 if i != 0]
a2 = len(liste2)
print(1000 - a2, " normalement on devrait tourver la même chose, donc on vérifie")

liste1 = [i for i in liste1 if i!= -1]
b1 = len(liste1)
print("Il y a : ", a1 - b1, " entrées qui n'ont pas d'adversaires avec la méthode de l'iterative gradient")
liste2 = [i for i in liste2 if i != -1]
b2 = len(liste2)
print("Il y a : ", a2 - b2, " entrées qui n'ont pas d'adversaires avec la méthode de l'article Intriguing properties")

liste1 = [i for i in liste1 if not(np.isnan(i))]
print("Il y a : ", b1 - len(liste1), " entrées qui admettent nan avec la méthode de l'iterative gradient")
liste2 = [i for i in liste2 if not(np.isnan(i))]
print("Il y a : ", b2 - len(liste2), " entrées qui admettent nan avec la méthode de l'article Intriguing properties")

distortion_moyenne1 = sum([np.sqrt(i) for i in liste1])/len(liste1)
distortion_moyenne2 = sum([np.sqrt(i) for i in liste2])/len(liste2)

print("La distortion moyenne dans le cas de l'iterative gradient est : ", distortion_moyenne1)
print("La distortion moyenne dans le cas de Intriguing properties est : ", distortion_moyenne2)


#######################
# Résultats des tests #
#######################

# On tire au sort 1000 images de la BDD test et le but est de générer un adveraire à partir de ces images

# 1 - Test FC100_100_10, nbre_entrainement = 10
###############################################

# Entrainement
# Entrainement 0001, loss: 0.5926, accuracy: 84.7067%, temps:  1.6405
# Entrainement 0002, loss: 0.2257, accuracy: 93.4367%, temps:  1.3137
# Entrainement 0003, loss: 0.1667, accuracy: 95.1250%, temps:  1.3234
# Entrainement 0004, loss: 0.1314, accuracy: 96.2100%, temps:  1.3471
# Entrainement 0005, loss: 0.1070, accuracy: 96.9050%, temps:  1.3269
# Entrainement 0006, loss: 0.0889, accuracy: 97.4000%, temps:  1.3318
# Entrainement 0007, loss: 0.0749, accuracy: 97.8250%, temps:  1.3188
# Entrainement 0008, loss: 0.0636, accuracy: 98.1800%, temps:  1.3266
# Entrainement 0009, loss: 0.0543, accuracy: 98.4733%, temps:  1.3429
# Entrainement 0010, loss: 0.0463, accuracy: 98.7367%, temps:  1.3625
# Jeu de test
# Loss: 0.0910, accuracy: 97.3500%, temps:  0.0214
# Il y a :  25  prédictions fausses au départ, donc inutile de leur chercher un adversaire
# 25  normalement on devrait tourver la même chose, donc on vérifie
# Il y a :  0  entrées qui n'ont pas d'adversaires avec la méthode de l'iterative gradient
# Il y a :  0  entrées qui n'ont pas d'adversaires avec la méthode de l'article Intriguing properties
# Il y a :  0  entrées qui admettent nan avec la méthode de l'iterative gradient
# Il y a :  0  entrées qui admettent nan avec la méthode de l'article Intriguing properties
# La distortion moyenne dans le cas de l'iterative gradient est :  0.02887265912340715
# La distortion moyenne dans le cas de Intriguing properties est :  0.029484260206628785

# 2 - Test FC100_100_10, nbre_entrainement = 50
###############################################


# Entrainement
# Entrainement 0001, loss: 0.6084, accuracy: 84.6483%, temps:  1.6991
# Entrainement 0002, loss: 0.2276, accuracy: 93.4017%, temps:  1.3400
# Entrainement 0003, loss: 0.1712, accuracy: 95.0017%, temps:  1.3413
# Entrainement 0004, loss: 0.1361, accuracy: 95.9700%, temps:  1.4204
# Entrainement 0005, loss: 0.1110, accuracy: 96.7500%, temps:  1.4467
# Entrainement 0006, loss: 0.0923, accuracy: 97.3233%, temps:  1.3654
# Entrainement 0007, loss: 0.0778, accuracy: 97.7800%, temps:  1.3594
# Entrainement 0008, loss: 0.0659, accuracy: 98.1350%, temps:  1.3808
# Entrainement 0009, loss: 0.0559, accuracy: 98.4117%, temps:  1.4083
# Entrainement 0010, loss: 0.0475, accuracy: 98.6833%, temps:  1.3427
# Entrainement 0011, loss: 0.0402, accuracy: 98.9200%, temps:  1.3879
# Entrainement 0012, loss: 0.0338, accuracy: 99.1533%, temps:  1.3722
# Entrainement 0013, loss: 0.0282, accuracy: 99.3583%, temps:  1.3256
# Entrainement 0014, loss: 0.0233, accuracy: 99.5133%, temps:  1.3404
# Entrainement 0015, loss: 0.0192, accuracy: 99.6300%, temps:  1.3645
# Entrainement 0016, loss: 0.0157, accuracy: 99.7217%, temps:  1.3355
# Entrainement 0017, loss: 0.0127, accuracy: 99.8100%, temps:  1.3545
# Entrainement 0018, loss: 0.0102, accuracy: 99.8700%, temps:  1.3515
# Entrainement 0019, loss: 0.0081, accuracy: 99.9000%, temps:  1.3270
# Entrainement 0020, loss: 0.0063, accuracy: 99.9350%, temps:  1.3349
# Entrainement 0021, loss: 0.0048, accuracy: 99.9583%, temps:  1.3239
# Entrainement 0022, loss: 0.0038, accuracy: 99.9700%, temps:  1.3261
# Entrainement 0023, loss: 0.0041, accuracy: 99.9450%, temps:  1.3571
# Entrainement 0024, loss: 0.0039, accuracy: 99.9450%, temps:  1.3358
# Entrainement 0025, loss: 0.0033, accuracy: 99.9500%, temps:  1.3322
# Entrainement 0026, loss: 0.0037, accuracy: 99.9217%, temps:  1.3569
# Entrainement 0027, loss: 0.0025, accuracy: 99.9683%, temps:  1.3323
# Entrainement 0028, loss: 0.0024, accuracy: 99.9517%, temps:  1.3650
# Entrainement 0029, loss: 0.0032, accuracy: 99.9267%, temps:  1.3335
# Entrainement 0030, loss: 0.0018, accuracy: 99.9767%, temps:  1.3296
# Entrainement 0031, loss: 0.0014, accuracy: 99.9783%, temps:  1.3765
# Entrainement 0032, loss: 0.0019, accuracy: 99.9550%, temps:  1.3430
# Entrainement 0033, loss: 0.0020, accuracy: 99.9583%, temps:  1.3657
# Entrainement 0034, loss: 0.0014, accuracy: 99.9733%, temps:  1.3415
# Entrainement 0035, loss: 0.0028, accuracy: 99.9200%, temps:  1.3498
# Entrainement 0036, loss: 0.0018, accuracy: 99.9567%, temps:  1.3370
# Entrainement 0037, loss: 0.0015, accuracy: 99.9767%, temps:  1.3199
# Entrainement 0038, loss: 0.0018, accuracy: 99.9617%, temps:  1.3476
# Entrainement 0039, loss: 0.0019, accuracy: 99.9433%, temps:  1.3332
# Entrainement 0040, loss: 0.0027, accuracy: 99.9200%, temps:  1.3622
# Entrainement 0041, loss: 0.0005, accuracy: 99.9933%, temps:  1.3499
# Entrainement 0042, loss: 0.0002, accuracy: 100.0000%, temps:  1.3472
# Entrainement 0043, loss: 0.0001, accuracy: 100.0000%, temps:  1.3373
# Entrainement 0044, loss: 0.0001, accuracy: 100.0000%, temps:  1.3517
# Entrainement 0045, loss: 0.0001, accuracy: 100.0000%, temps:  1.3290
# Entrainement 0046, loss: 0.0001, accuracy: 100.0000%, temps:  1.3607
# Entrainement 0047, loss: 0.0000, accuracy: 100.0000%, temps:  1.3230
# Entrainement 0048, loss: 0.0000, accuracy: 100.0000%, temps:  1.3590
# Entrainement 0049, loss: 0.0000, accuracy: 100.0000%, temps:  1.3623
# Entrainement 0050, loss: 0.0000, accuracy: 100.0000%, temps:  1.3571
# Jeu de test
# Loss: 0.1399, accuracy: 97.9100%, temps:  0.0272
# Il y a :  20  prédictions fausses au départ, donc inutile de leur chercher un adversaire
# 20  normalement on devrait tourver la même chose, donc on vérifie
# Il y a :  76  entrées qui n'ont pas d'adversaires avec la méthode de l'iterative gradient
# Il y a :  766  entrées qui n'ont pas d'adversaires avec la méthode de l'article Intriguing properties
# Il y a :  690  entrées qui admettent nan avec la méthode de l'iterative gradient
# Il y a :  0  entrées qui admettent nan avec la méthode de l'article Intriguing properties
# La distortion moyenne dans le cas de l'iterative gradient est :  0.018721596546749646
# La distortion moyenne dans le cas de Intriguing properties est :  0.012283262739712432

# 3 - Test FC100_100_10, nbre_entrainement = 100
################################################

# Entrainement
# Entrainement 0001, loss: 0.5750, accuracy: 85.2833%, temps:  1.6905
# Entrainement 0002, loss: 0.2292, accuracy: 93.3467%, temps:  1.3731
# Entrainement 0003, loss: 0.1711, accuracy: 94.9667%, temps:  1.3281
# Entrainement 0004, loss: 0.1346, accuracy: 96.0317%, temps:  1.4100
# Entrainement 0005, loss: 0.1090, accuracy: 96.8483%, temps:  1.3993
# Entrainement 0006, loss: 0.0901, accuracy: 97.3667%, temps:  1.4112
# Entrainement 0007, loss: 0.0753, accuracy: 97.8050%, temps:  1.3477
# Entrainement 0008, loss: 0.0633, accuracy: 98.2067%, temps:  1.3499
# Entrainement 0009, loss: 0.0533, accuracy: 98.5100%, temps:  1.3308
# Entrainement 0010, loss: 0.0449, accuracy: 98.7900%, temps:  1.3329
# Entrainement 0011, loss: 0.0377, accuracy: 99.0583%, temps:  1.3432
# Entrainement 0012, loss: 0.0315, accuracy: 99.2583%, temps:  1.3282
# Entrainement 0013, loss: 0.0262, accuracy: 99.4250%, temps:  1.3270
# Entrainement 0014, loss: 0.0217, accuracy: 99.5667%, temps:  1.3560
# Entrainement 0015, loss: 0.0178, accuracy: 99.6667%, temps:  1.3215
# Entrainement 0016, loss: 0.0145, accuracy: 99.7583%, temps:  1.3444
# Entrainement 0017, loss: 0.0116, accuracy: 99.8317%, temps:  1.3140
# Entrainement 0018, loss: 0.0092, accuracy: 99.8917%, temps:  1.3268
# Entrainement 0019, loss: 0.0073, accuracy: 99.9250%, temps:  1.3021
# Entrainement 0020, loss: 0.0057, accuracy: 99.9433%, temps:  1.3556
# Entrainement 0021, loss: 0.0045, accuracy: 99.9567%, temps:  1.3406
# Entrainement 0022, loss: 0.0046, accuracy: 99.9383%, temps:  1.3295
# Entrainement 0023, loss: 0.0050, accuracy: 99.9033%, temps:  1.3348
# Entrainement 0024, loss: 0.0041, accuracy: 99.9283%, temps:  1.3494
# Entrainement 0025, loss: 0.0030, accuracy: 99.9550%, temps:  1.3093
# Entrainement 0026, loss: 0.0025, accuracy: 99.9600%, temps:  1.3374
# Entrainement 0027, loss: 0.0027, accuracy: 99.9533%, temps:  1.3315
# Entrainement 0028, loss: 0.0029, accuracy: 99.9350%, temps:  1.3231
# Entrainement 0029, loss: 0.0023, accuracy: 99.9667%, temps:  1.3510
# Entrainement 0030, loss: 0.0038, accuracy: 99.8867%, temps:  1.3160
# Entrainement 0031, loss: 0.0016, accuracy: 99.9817%, temps:  1.3336
# Entrainement 0032, loss: 0.0017, accuracy: 99.9767%, temps:  1.3296
# Entrainement 0033, loss: 0.0039, accuracy: 99.8867%, temps:  1.3296
# Entrainement 0034, loss: 0.0027, accuracy: 99.9200%, temps:  1.3332
# Entrainement 0035, loss: 0.0011, accuracy: 99.9800%, temps:  1.3408
# Entrainement 0036, loss: 0.0026, accuracy: 99.9283%, temps:  1.3437
# Entrainement 0037, loss: 0.0009, accuracy: 99.9867%, temps:  1.3333
# Entrainement 0038, loss: 0.0016, accuracy: 99.9567%, temps:  1.3509
# Entrainement 0039, loss: 0.0027, accuracy: 99.9250%, temps:  1.3385
# Entrainement 0040, loss: 0.0014, accuracy: 99.9700%, temps:  1.3394
# Entrainement 0041, loss: 0.0004, accuracy: 99.9983%, temps:  1.3489
# Entrainement 0042, loss: 0.0002, accuracy: 100.0000%, temps:  1.3285
# Entrainement 0043, loss: 0.0001, accuracy: 100.0000%, temps:  1.3214
# Entrainement 0044, loss: 0.0001, accuracy: 100.0000%, temps:  1.3408
# Entrainement 0045, loss: 0.0001, accuracy: 100.0000%, temps:  1.3614
# Entrainement 0046, loss: 0.0001, accuracy: 100.0000%, temps:  1.3404
# Entrainement 0047, loss: 0.0000, accuracy: 100.0000%, temps:  1.3123
# Entrainement 0048, loss: 0.0000, accuracy: 100.0000%, temps:  1.3480
# Entrainement 0049, loss: 0.0006, accuracy: 99.9850%, temps:  1.3413
# Entrainement 0050, loss: 0.0082, accuracy: 99.7683%, temps:  1.3384
# Entrainement 0051, loss: 0.0005, accuracy: 99.9900%, temps:  1.3341
# Entrainement 0052, loss: 0.0001, accuracy: 100.0000%, temps:  1.3581
# Entrainement 0053, loss: 0.0001, accuracy: 100.0000%, temps:  1.3412
# Entrainement 0054, loss: 0.0000, accuracy: 100.0000%, temps:  1.3304
# Entrainement 0055, loss: 0.0000, accuracy: 100.0000%, temps:  1.3318
# Entrainement 0056, loss: 0.0000, accuracy: 100.0000%, temps:  1.3517
# Entrainement 0057, loss: 0.0000, accuracy: 100.0000%, temps:  1.3871
# Entrainement 0058, loss: 0.0000, accuracy: 100.0000%, temps:  1.3172
# Entrainement 0059, loss: 0.0000, accuracy: 100.0000%, temps:  1.3220
# Entrainement 0060, loss: 0.0000, accuracy: 100.0000%, temps:  1.3439
# Entrainement 0061, loss: 0.0000, accuracy: 100.0000%, temps:  1.3249
# Entrainement 0062, loss: 0.0000, accuracy: 100.0000%, temps:  1.3250
# Entrainement 0063, loss: 0.0000, accuracy: 100.0000%, temps:  1.3315
# Entrainement 0064, loss: 0.0000, accuracy: 100.0000%, temps:  1.3341
# Entrainement 0065, loss: 0.0000, accuracy: 100.0000%, temps:  1.3445
# Entrainement 0066, loss: 0.0080, accuracy: 99.7883%, temps:  1.3432
# Entrainement 0067, loss: 0.0009, accuracy: 99.9767%, temps:  1.3228
# Entrainement 0068, loss: 0.0004, accuracy: 99.9917%, temps:  1.3280
# Entrainement 0069, loss: 0.0001, accuracy: 100.0000%, temps:  1.3552
# Entrainement 0070, loss: 0.0038, accuracy: 99.8733%, temps:  1.3507
# Entrainement 0071, loss: 0.0012, accuracy: 99.9550%, temps:  1.3290
# Entrainement 0072, loss: 0.0005, accuracy: 99.9817%, temps:  1.3550
# Entrainement 0073, loss: 0.0015, accuracy: 99.9533%, temps:  1.3547
# Entrainement 0074, loss: 0.0026, accuracy: 99.9000%, temps:  1.3303
# Entrainement 0075, loss: 0.0005, accuracy: 99.9817%, temps:  1.3202
# Entrainement 0076, loss: 0.0001, accuracy: 99.9983%, temps:  1.3352
# Entrainement 0077, loss: 0.0000, accuracy: 100.0000%, temps:  1.3336
# Entrainement 0078, loss: 0.0000, accuracy: 100.0000%, temps:  1.3506
# Entrainement 0079, loss: 0.0000, accuracy: 100.0000%, temps:  1.3640
# Entrainement 0080, loss: 0.0000, accuracy: 100.0000%, temps:  1.3442
# Entrainement 0081, loss: 0.0000, accuracy: 100.0000%, temps:  1.3278
# Entrainement 0082, loss: 0.0000, accuracy: 100.0000%, temps:  1.3116
# Entrainement 0083, loss: 0.0000, accuracy: 100.0000%, temps:  1.3118
# Entrainement 0084, loss: 0.0000, accuracy: 100.0000%, temps:  1.3600
# Entrainement 0085, loss: 0.0000, accuracy: 100.0000%, temps:  1.3474
# Entrainement 0086, loss: 0.0000, accuracy: 100.0000%, temps:  1.3175
# Entrainement 0087, loss: 0.0000, accuracy: 100.0000%, temps:  1.3363
# Entrainement 0088, loss: 0.0000, accuracy: 100.0000%, temps:  1.3262
# Entrainement 0089, loss: 0.0000, accuracy: 100.0000%, temps:  1.3613
# Entrainement 0090, loss: 0.0000, accuracy: 100.0000%, temps:  1.3349
# Entrainement 0091, loss: 0.0000, accuracy: 100.0000%, temps:  1.3500
# Entrainement 0092, loss: 0.0000, accuracy: 100.0000%, temps:  1.3277
# Entrainement 0093, loss: 0.0000, accuracy: 100.0000%, temps:  1.3190
# Entrainement 0094, loss: 0.0000, accuracy: 100.0000%, temps:  1.3142
# Entrainement 0095, loss: 0.0000, accuracy: 100.0000%, temps:  1.3467
# Entrainement 0096, loss: 0.0000, accuracy: 100.0000%, temps:  1.3490
# Entrainement 0097, loss: 0.0000, accuracy: 100.0000%, temps:  1.3297
# Entrainement 0098, loss: 0.0000, accuracy: 100.0000%, temps:  1.3216
# Entrainement 0099, loss: 0.0000, accuracy: 100.0000%, temps:  1.3374
# Entrainement 0100, loss: 0.0000, accuracy: 100.0000%, temps:  1.3275
# Jeu de test
# Loss: 0.1772, accuracy: 97.8000%, temps:  0.0232
# Il y a :  17  prédictions fausses au départ, donc inutile de leur chercher un adversaire
# 17  normalement on devrait tourver la même chose, donc on vérifie
# Il y a :  97  entrées qui n'ont pas d'adversaires avec la méthode de l'iterative gradient
# Il y a :  890  entrées qui n'ont pas d'adversaires avec la méthode de l'article Intriguing properties
# Il y a :  793  entrées qui admettent nan avec la méthode de l'iterative gradient
# Il y a :  0  entrées qui admettent nan avec la méthode de l'article Intriguing properties
# La distortion moyenne dans le cas de l'iterative gradient est :  0.024618180003017187
# La distortion moyenne dans le cas de Intriguing properties est :  0.0076556620886393135


# 4 - Test FC200_200_10, nbre_entrainement = 10
###############################################

# Entrainement
# Entrainement 0001, loss: 0.4750, accuracy: 87.0850%, temps:  2.2304
# Entrainement 0002, loss: 0.2053, accuracy: 93.8867%, temps:  1.8981
# Entrainement 0003, loss: 0.1504, accuracy: 95.5017%, temps:  1.9202
# Entrainement 0004, loss: 0.1153, accuracy: 96.5467%, temps:  1.9172
# Entrainement 0005, loss: 0.0899, accuracy: 97.3083%, temps:  1.9296
# Entrainement 0006, loss: 0.0708, accuracy: 97.8767%, temps:  1.9179
# Entrainement 0007, loss: 0.0558, accuracy: 98.3567%, temps:  1.9116
# Entrainement 0008, loss: 0.0439, accuracy: 98.7533%, temps:  1.9286
# Entrainement 0009, loss: 0.0343, accuracy: 99.0500%, temps:  1.9108
# Entrainement 0010, loss: 0.0265, accuracy: 99.3217%, temps:  1.9200
# Jeu de test
# Loss: 0.0859, accuracy: 97.4900%, temps:  0.0418
# Il y a :  24  prédictions fausses au départ, donc inutile de leur chercher un adversaire
# 24  normalement on devrait tourver la même chose, donc on vérifie
# Il y a :  0  entrées qui n'ont pas d'adversaires avec la méthode de l'iterative gradient
# Il y a :  3  entrées qui n'ont pas d'adversaires avec la méthode de l'article Intriguing properties
# Il y a :  3  entrées qui admettent nan avec la méthode de l'iterative gradient
# Il y a :  0  entrées qui admettent nan avec la méthode de l'article Intriguing properties
# La distortion moyenne dans le cas de l'iterative gradient est :  0.031906340444601215
# La distortion moyenne dans le cas de Intriguing properties est :  0.032279533620081007


# 5 - Test FC200_200_10, nbre_entrainement = 50
###############################################

# Entrainement
# Entrainement 0001, loss: 0.4815, accuracy: 86.9533%, temps:  2.2339
# Entrainement 0002, loss: 0.2054, accuracy: 93.8983%, temps:  1.9018
# Entrainement 0003, loss: 0.1483, accuracy: 95.5233%, temps:  1.9261
# Entrainement 0004, loss: 0.1123, accuracy: 96.6683%, temps:  1.9235
# Entrainement 0005, loss: 0.0872, accuracy: 97.4350%, temps:  1.9121
# Entrainement 0006, loss: 0.0688, accuracy: 98.0200%, temps:  1.9198
# Entrainement 0007, loss: 0.0544, accuracy: 98.4250%, temps:  1.9267
# Entrainement 0008, loss: 0.0427, accuracy: 98.7817%, temps:  1.9097
# Entrainement 0009, loss: 0.0330, accuracy: 99.0750%, temps:  1.9312
# Entrainement 0010, loss: 0.0250, accuracy: 99.3750%, temps:  1.9295
# Entrainement 0011, loss: 0.0189, accuracy: 99.5917%, temps:  1.9291
# Entrainement 0012, loss: 0.0139, accuracy: 99.7083%, temps:  1.9286
# Entrainement 0013, loss: 0.0102, accuracy: 99.8067%, temps:  1.9161
# Entrainement 0014, loss: 0.0083, accuracy: 99.8483%, temps:  1.9194
# Entrainement 0015, loss: 0.0076, accuracy: 99.8283%, temps:  1.9333
# Entrainement 0016, loss: 0.0062, accuracy: 99.8483%, temps:  1.9393
# Entrainement 0017, loss: 0.0068, accuracy: 99.8183%, temps:  1.9366
# Entrainement 0018, loss: 0.0043, accuracy: 99.8867%, temps:  1.9288
# Entrainement 0019, loss: 0.0041, accuracy: 99.8900%, temps:  1.9377
# Entrainement 0020, loss: 0.0039, accuracy: 99.8850%, temps:  1.9236
# Entrainement 0021, loss: 0.0039, accuracy: 99.8783%, temps:  1.9276
# Entrainement 0022, loss: 0.0052, accuracy: 99.8267%, temps:  1.9307
# Entrainement 0023, loss: 0.0032, accuracy: 99.8983%, temps:  1.9316
# Entrainement 0024, loss: 0.0029, accuracy: 99.9183%, temps:  1.9284
# Entrainement 0025, loss: 0.0039, accuracy: 99.8817%, temps:  1.9316
# Entrainement 0026, loss: 0.0011, accuracy: 99.9817%, temps:  1.9254
# Entrainement 0027, loss: 0.0056, accuracy: 99.8083%, temps:  1.9356
# Entrainement 0028, loss: 0.0015, accuracy: 99.9700%, temps:  1.9439
# Entrainement 0029, loss: 0.0009, accuracy: 99.9750%, temps:  1.9371
# Entrainement 0030, loss: 0.0055, accuracy: 99.8367%, temps:  1.9532
# Entrainement 0031, loss: 0.0004, accuracy: 99.9967%, temps:  1.9348
# Entrainement 0032, loss: 0.0008, accuracy: 99.9817%, temps:  1.9132
# Entrainement 0033, loss: 0.0055, accuracy: 99.8133%, temps:  1.9260
# Entrainement 0034, loss: 0.0021, accuracy: 99.9300%, temps:  1.9311
# Entrainement 0035, loss: 0.0030, accuracy: 99.9183%, temps:  1.9259
# Entrainement 0036, loss: 0.0017, accuracy: 99.9600%, temps:  1.9253
# Entrainement 0037, loss: 0.0025, accuracy: 99.9267%, temps:  1.9275
# Entrainement 0038, loss: 0.0006, accuracy: 99.9867%, temps:  1.9281
# Entrainement 0039, loss: 0.0034, accuracy: 99.8950%, temps:  1.9357
# Entrainement 0040, loss: 0.0024, accuracy: 99.9217%, temps:  1.9318
# Entrainement 0041, loss: 0.0013, accuracy: 99.9633%, temps:  1.9398
# Entrainement 0042, loss: 0.0012, accuracy: 99.9650%, temps:  1.9590
# Entrainement 0043, loss: 0.0014, accuracy: 99.9567%, temps:  1.9300
# Entrainement 0044, loss: 0.0035, accuracy: 99.8800%, temps:  1.9238
# Entrainement 0045, loss: 0.0005, accuracy: 99.9850%, temps:  1.9177
# Entrainement 0046, loss: 0.0010, accuracy: 99.9750%, temps:  1.9332
# Entrainement 0047, loss: 0.0036, accuracy: 99.8633%, temps:  1.9467
# Entrainement 0048, loss: 0.0010, accuracy: 99.9733%, temps:  1.9378
# Entrainement 0049, loss: 0.0019, accuracy: 99.9383%, temps:  1.9559
# Entrainement 0050, loss: 0.0024, accuracy: 99.9083%, temps:  1.9244
# Jeu de test
# Loss: 0.1313, accuracy: 97.7400%, temps:  0.0377
# Il y a :  19  prédictions fausses au départ, donc inutile de leur chercher un adversaire
# 19  normalement on devrait tourver la même chose, donc on vérifie
# Il y a :  88  entrées qui n'ont pas d'adversaires avec la méthode de l'iterative gradient
# Il y a :  789  entrées qui n'ont pas d'adversaires avec la méthode de l'article Intriguing properties
# Il y a :  701  entrées qui admettent nan avec la méthode de l'iterative gradient
# Il y a :  0  entrées qui admettent nan avec la méthode de l'article Intriguing properties
# La distortion moyenne dans le cas de l'iterative gradient est :  0.01947929822684576
# La distortion moyenne dans le cas de Intriguing properties est :  0.01557716938107688

# 3 - Test FC200_200_10, nbre_entrainement = 100
################################################

# Entrainement
# Entrainement 0001, loss: 0.4805, accuracy: 86.9067%, temps:  2.3232
# Entrainement 0002, loss: 0.2069, accuracy: 93.8517%, temps:  1.9573
# Entrainement 0003, loss: 0.1509, accuracy: 95.4617%, temps:  2.0058
# Entrainement 0004, loss: 0.1146, accuracy: 96.5800%, temps:  1.9087
# Entrainement 0005, loss: 0.0888, accuracy: 97.3900%, temps:  1.9092
# Entrainement 0006, loss: 0.0697, accuracy: 97.9383%, temps:  1.9172
# Entrainement 0007, loss: 0.0548, accuracy: 98.3983%, temps:  1.9052
# Entrainement 0008, loss: 0.0430, accuracy: 98.7683%, temps:  1.9221
# Entrainement 0009, loss: 0.0336, accuracy: 99.0800%, temps:  1.8990
# Entrainement 0010, loss: 0.0260, accuracy: 99.3367%, temps:  1.9144
# Entrainement 0011, loss: 0.0199, accuracy: 99.5617%, temps:  1.9114
# Entrainement 0012, loss: 0.0152, accuracy: 99.6683%, temps:  1.9136
# Entrainement 0013, loss: 0.0115, accuracy: 99.7783%, temps:  1.9181
# Entrainement 0014, loss: 0.0088, accuracy: 99.8417%, temps:  1.9275
# Entrainement 0015, loss: 0.0083, accuracy: 99.7950%, temps:  1.9183
# Entrainement 0016, loss: 0.0077, accuracy: 99.8033%, temps:  1.9108
# Entrainement 0017, loss: 0.0069, accuracy: 99.7883%, temps:  1.9219
# Entrainement 0018, loss: 0.0054, accuracy: 99.8467%, temps:  1.9161
# Entrainement 0019, loss: 0.0036, accuracy: 99.9183%, temps:  1.9281
# Entrainement 0020, loss: 0.0041, accuracy: 99.8750%, temps:  1.9202
# Entrainement 0021, loss: 0.0049, accuracy: 99.8467%, temps:  1.9172
# Entrainement 0022, loss: 0.0040, accuracy: 99.8850%, temps:  1.9176
# Entrainement 0023, loss: 0.0020, accuracy: 99.9583%, temps:  1.9400
# Entrainement 0024, loss: 0.0044, accuracy: 99.8500%, temps:  2.0052
# Entrainement 0025, loss: 0.0025, accuracy: 99.9367%, temps:  1.9763
# Entrainement 0026, loss: 0.0027, accuracy: 99.9300%, temps:  1.9555
# Entrainement 0027, loss: 0.0024, accuracy: 99.9317%, temps:  1.9307
# Entrainement 0028, loss: 0.0052, accuracy: 99.8283%, temps:  1.9308
# Entrainement 0029, loss: 0.0018, accuracy: 99.9450%, temps:  1.9882
# Entrainement 0030, loss: 0.0033, accuracy: 99.9050%, temps:  1.9342
# Entrainement 0031, loss: 0.0013, accuracy: 99.9683%, temps:  1.9301
# Entrainement 0032, loss: 0.0047, accuracy: 99.8517%, temps:  1.9357
# Entrainement 0033, loss: 0.0014, accuracy: 99.9617%, temps:  1.9187
# Entrainement 0034, loss: 0.0010, accuracy: 99.9767%, temps:  1.9372
# Entrainement 0035, loss: 0.0031, accuracy: 99.9117%, temps:  1.9157
# Entrainement 0036, loss: 0.0022, accuracy: 99.9267%, temps:  1.9324
# Entrainement 0037, loss: 0.0011, accuracy: 99.9683%, temps:  1.9299
# Entrainement 0038, loss: 0.0025, accuracy: 99.9167%, temps:  1.9197
# Entrainement 0039, loss: 0.0031, accuracy: 99.8900%, temps:  1.9253
# Entrainement 0040, loss: 0.0023, accuracy: 99.9233%, temps:  1.9198
# Entrainement 0041, loss: 0.0019, accuracy: 99.9367%, temps:  1.9308
# Entrainement 0042, loss: 0.0010, accuracy: 99.9700%, temps:  1.9140
# Entrainement 0043, loss: 0.0019, accuracy: 99.9450%, temps:  1.9127
# Entrainement 0044, loss: 0.0023, accuracy: 99.9217%, temps:  1.9284
# Entrainement 0045, loss: 0.0004, accuracy: 99.9933%, temps:  1.9256
# Entrainement 0046, loss: 0.0001, accuracy: 100.0000%, temps:  1.9236
# Entrainement 0047, loss: 0.0000, accuracy: 100.0000%, temps:  1.9202
# Entrainement 0048, loss: 0.0000, accuracy: 100.0000%, temps:  1.9280
# Entrainement 0049, loss: 0.0000, accuracy: 100.0000%, temps:  1.9121
# Entrainement 0050, loss: 0.0000, accuracy: 100.0000%, temps:  1.9045
# Entrainement 0051, loss: 0.0000, accuracy: 100.0000%, temps:  1.9297
# Entrainement 0052, loss: 0.0000, accuracy: 100.0000%, temps:  1.9299
# Entrainement 0053, loss: 0.0000, accuracy: 100.0000%, temps:  1.9104
# Entrainement 0054, loss: 0.0000, accuracy: 100.0000%, temps:  1.9201
# Entrainement 0055, loss: 0.0000, accuracy: 100.0000%, temps:  1.9196
# Entrainement 0056, loss: 0.0000, accuracy: 100.0000%, temps:  1.9226
# Entrainement 0057, loss: 0.0000, accuracy: 100.0000%, temps:  1.9287
# Entrainement 0058, loss: 0.0000, accuracy: 100.0000%, temps:  1.9227
# Entrainement 0059, loss: 0.0000, accuracy: 100.0000%, temps:  1.9229
# Entrainement 0060, loss: 0.0000, accuracy: 100.0000%, temps:  1.9251
# Entrainement 0061, loss: 0.0000, accuracy: 100.0000%, temps:  1.9070
# Entrainement 0062, loss: 0.0000, accuracy: 100.0000%, temps:  1.9108
# Entrainement 0063, loss: 0.0000, accuracy: 100.0000%, temps:  1.9287
# Entrainement 0064, loss: 0.0000, accuracy: 100.0000%, temps:  1.9214
# Entrainement 0065, loss: 0.0000, accuracy: 100.0000%, temps:  1.9302
# Entrainement 0066, loss: 0.0000, accuracy: 100.0000%, temps:  1.9316
# Entrainement 0067, loss: 0.0000, accuracy: 100.0000%, temps:  1.9403
# Entrainement 0068, loss: 0.0000, accuracy: 100.0000%, temps:  1.9264
# Entrainement 0069, loss: 0.0000, accuracy: 100.0000%, temps:  1.9221
# Entrainement 0070, loss: 0.0000, accuracy: 100.0000%, temps:  1.9209
# Entrainement 0071, loss: 0.0000, accuracy: 100.0000%, temps:  1.9240
# Entrainement 0072, loss: 0.0000, accuracy: 100.0000%, temps:  1.9350
# Entrainement 0073, loss: 0.0000, accuracy: 100.0000%, temps:  1.9349
# Entrainement 0074, loss: 0.0000, accuracy: 100.0000%, temps:  1.9258
# Entrainement 0075, loss: 0.0000, accuracy: 100.0000%, temps:  1.9217
# Entrainement 0076, loss: 0.0000, accuracy: 100.0000%, temps:  1.9226
# Entrainement 0077, loss: 0.0000, accuracy: 100.0000%, temps:  1.9282
# Entrainement 0078, loss: 0.0000, accuracy: 100.0000%, temps:  1.9196
# Entrainement 0079, loss: 0.0000, accuracy: 100.0000%, temps:  1.9357
# Entrainement 0080, loss: 0.0000, accuracy: 100.0000%, temps:  1.9169
# Entrainement 0081, loss: 0.0000, accuracy: 100.0000%, temps:  1.9332
# Entrainement 0082, loss: 0.0000, accuracy: 100.0000%, temps:  1.9315
# Entrainement 0083, loss: 0.0000, accuracy: 100.0000%, temps:  1.9298
# Entrainement 0084, loss: 0.0000, accuracy: 100.0000%, temps:  1.9276
# Entrainement 0085, loss: 0.0000, accuracy: 100.0000%, temps:  1.9190
# Entrainement 0086, loss: 0.0000, accuracy: 100.0000%, temps:  1.9260
# Entrainement 0087, loss: 0.0000, accuracy: 100.0000%, temps:  1.9285
# Entrainement 0088, loss: 0.0000, accuracy: 100.0000%, temps:  1.9192
# Entrainement 0089, loss: 0.0000, accuracy: 100.0000%, temps:  1.9304
# Entrainement 0090, loss: 0.0000, accuracy: 100.0000%, temps:  2.0211
# Entrainement 0091, loss: 0.0000, accuracy: 100.0000%, temps:  1.9687
# Entrainement 0092, loss: 0.0000, accuracy: 100.0000%, temps:  1.9255
# Entrainement 0093, loss: 0.0000, accuracy: 100.0000%, temps:  1.9232
# Entrainement 0094, loss: 0.0000, accuracy: 100.0000%, temps:  1.9167
# Entrainement 0095, loss: 0.0000, accuracy: 100.0000%, temps:  1.9333
# Entrainement 0096, loss: 0.0000, accuracy: 100.0000%, temps:  1.9396
# Entrainement 0097, loss: 0.0000, accuracy: 100.0000%, temps:  1.9339
# Entrainement 0098, loss: 0.0000, accuracy: 100.0000%, temps:  1.9219
# Entrainement 0099, loss: 0.0000, accuracy: 100.0000%, temps:  1.9175
# Entrainement 0100, loss: 0.0000, accuracy: 100.0000%, temps:  1.9302
# Jeu de test
# Loss: 0.1500, accuracy: 98.2800%, temps:  0.0423
# Il y a :  14  prédictions fausses au départ, donc inutile de leur chercher un adversaire
# 14  normalement on devrait tourver la même chose, donc on vérifie
# Il y a :  105  entrées qui n'ont pas d'adversaires avec la méthode de l'iterative gradient
# Il y a :  913  entrées qui n'ont pas d'adversaires avec la méthode de l'article Intriguing properties
# Il y a :  808  entrées qui admettent nan avec la méthode de l'iterative gradient
# Il y a :  0  entrées qui admettent nan avec la méthode de l'article Intriguing properties
# La distortion moyenne dans le cas de l'iterative gradient est :  0.02005558339751338
# La distortion moyenne dans le cas de Intriguing properties est :  0.009674706347198631