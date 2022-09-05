from __future__ import absolute_import, division, print_function, unicode_literals
import os



import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import numpy as np
import sklearn
import pprint
import json

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from numpy import genfromtxt
from sklearn.metrics import classification_report, confusion_matrix

from art.attacks.poisoning.perturbations.image_perturbations import add_pattern_bd, add_single_bd
from art.estimators.classification import KerasClassifier
from art.utils import load_mnist, preprocess
from art.defences.detector.poison import ActivationDefence

def load_data():
    path = '/content/drive/MyDrive/DE-Poise Capstone2/De-Pois-Code/mnist (5).npz'
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


def TrueAndGeneratorData(know_rate, epochs):
    (x, y), (X_test, y_test) = load_data()
    trainct = int(know_rate * y.shape[0])   
    train_data = x[0:trainct,:,:]
    train_label = y[0:trainct]
    generator_size = len(y) - trainct

    print(generator_size,epochs)
    
    if (os.path.exists("Generator_data_%d_%d.npy"%(generator_size, epochs))):
        G_data = np.load("Generator_data_%d_%d.npy"%(generator_size, epochs))
        G_label = np.load("Generator_label_%d_%d.npy"%(generator_size, epochs))
    else:
        from generator_CGAN_authen import CGAN_data_loss
        CGAN_data_loss(know_rate, epochs)
        
        G_data = np.load("/content/Generate_data_%d_%d.npy"%(generator_size, epochs))
        G_label = np.load("/content/Generator_label_%d_%d.npy"%(generator_size, epochs))
    
    Max = np.max(G_data)
    Min = np.min(G_data)
    G_data = (G_data - Min) / (Max - Min)
    G_data = G_data * (1 * (G_data > 0.3))
    xx = np.concatenate((train_data/255,np.squeeze(G_data)), axis = 0)
    yy = np.concatenate((train_label,np.squeeze(G_label)), axis =0) # 0 ~ 1
    np.save("saved_TrueAndGeneratorData.npy", xx)
    np.save("saved_TrueAndGeneratorLabel.npy", yy)
    
    print('CGAN_loss True And Generator Data saved!')
        

def poi_data(poison_rate):   
    poison_number = int(poison_rate * 50000)
    poisoned_x_data = genfromtxt("/content/drive/MyDrive/DE-Poise Capstone2/De-Pois-Code/p_images.csv", encoding= 'unicode_escape')
    poisoned_y_data = genfromtxt("/content/drive/MyDrive/DE-Poise Capstone2/De-Pois-Code/p_labels.csv", encoding= 'unicode_escape')
    poisoned_x_data = poisoned_x_data * 255
    poisoned_x_data = np.arange(47040000).reshape(60000,1,28,28 )
    poisoned_x_data = (poisoned_x_data.astype(np.float32) - 127.5) / 127.5
    
    return poison_number, poisoned_x_data, poisoned_y_data


def count_score(validity, poison_number):
    label_poisoned_real = np.ones(validity.shape[0])
    label_poisoned_real[-poison_number:] = 0
    
    z_scores = np.mean(validity[0:-poison_number]) - np.std(validity[0:-poison_number])
    f_poisoned_data = np.where(validity<=z_scores)[0]
    
    label_poisoned_fake = np.ones(validity.shape[0])
    label_poisoned_fake[f_poisoned_data] = 0
    
    P = sklearn.metrics.precision_score(label_poisoned_real,label_poisoned_fake.astype(float),average='binary')
    R = sklearn.metrics.recall_score(label_poisoned_real,label_poisoned_fake.astype(float),average='binary')
    F1 = (2 * P * R) / (P + R)
    acc = sklearn.metrics.accuracy_score(label_poisoned_real,label_poisoned_fake.astype(float))
    
    return F1, P, R, acc



def modle_defense(poi_rate,  epochs):

    (X_train, y_train), (_, _) = load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=1)
    
    poison_number, poisoned_x_data, poisoned_y_data = poi_data(poi_rate)
    
    x_poisoned_raw_test = np.concatenate((X_train[0:50000],poisoned_x_data))   
    y_poisoned_raw_test = np.concatenate((y_train[0:50000],poisoned_y_data))
    
    from mimic_model_construction import CWGANGP
    batch_size = 32
    sample_interval = 100
    will_load_model = False
    
    start = time.perf_counter()
    wgan = CWGANGP(epochs, batch_size, sample_interval)
    wgan.train(will_load_model)
    end = time.perf_counter()
    print('Running time for train WGANGP: %s Seconds'%(end-start))

    D_poi = wgan.discriminate_img(x_poisoned_raw_test, y_poisoned_raw_test)
    
    D_poi = D_poi.flatten()
    cm =confusion_matrix(y_train,y_poisoned_raw_test[0:60000])
    print(cm)

    plt.title('CWGANGP D_value')  
    #plt.hist(D_poi[0:50000],bins = 100,color = 'b')
    #plt.hist(D_poi[50000:],bins = 100,color = 'r')
    plt.show()
    

    
    F1, P, R, acc = count_score(D_poi, poison_number)
    print('Accuracy of De-pois:')
    print(acc)
    print('Precision of De-pois：')
    print(P)
    print('Recall of De-pois：')
    print(R)
    print('F1 of De-pois：')
    print(F1)
    
    return None



    
if __name__ == '__main__':
    poi_rate = 0.30
    know_rate = 0.20

    TrueAndGeneratorData(know_rate, 2000)

    modle_defense(poi_rate, 2000)