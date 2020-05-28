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

# Entrainement 0041, loss: 0.0005, accuracy: 99.9933%, temps:  1.3499
# Entrainement 0042, loss: 0.0002, accuracy: 100.0000%, temps:  1.3472
# Entrainement 0043, loss: 0.0001, accuracy: 100.0000%, temps:  1.3373

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

# Entrainement 0041, loss: 0.0004, accuracy: 99.9983%, temps:  1.3489
# Entrainement 0042, loss: 0.0002, accuracy: 100.0000%, temps:  1.3285
# Entrainement 0043, loss: 0.0001, accuracy: 100.0000%, temps:  1.3214

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


# 6 - Test FC200_200_10, nbre_entrainement = 100
################################################

# Entrainement
# Entrainement 0001, loss: 0.4805, accuracy: 86.9067%, temps:  2.3232
# Entrainement 0002, loss: 0.2069, accuracy: 93.8517%, temps:  1.9573
# Entrainement 0003, loss: 0.1509, accuracy: 95.4617%, temps:  2.0058

# Entrainement 0045, loss: 0.0004, accuracy: 99.9933%, temps:  1.9256
# Entrainement 0046, loss: 0.0001, accuracy: 100.0000%, temps:  1.9236
# Entrainement 0047, loss: 0.0000, accuracy: 100.0000%, temps:  1.9202

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
