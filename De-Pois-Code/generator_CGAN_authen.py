import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.autograd import Variable
from tensorflow import keras
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding
from keras.layers import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K  
K.set_image_data_format('channels_first')
#'channels_first'` or `'channels_last'

import time

def build_lenet():

    # initialize the model
    model = Sequential()

    # first set of CONV => RELU => POOL
    model.add(Conv2D(20, 5, 5, padding="same",
                            input_shape=(1, 28, 28)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL
    model.add(Conv2D(50, 5, 5, padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    image = Input(shape=(1, 28, 28))

    features = model(image)

    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
    aux = Dense(10, activation='softmax', name='auxiliary')(features)

    return Model(image, aux)

def one_hot(labels, class_size): 
    targets = torch.zeros(labels.shape[0], class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return Variable(targets)


class CGAN():
    def __init__(self):

        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 10
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
            optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(512, input_dim=np.prod(self.img_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        flat_img = Flatten()(img)

        model_input = multiply([flat_img, label_embedding])
        validity = model(model_input)

        return Model([img, label], validity)

    def train(self,X_train, y_train, epochs, batch_size=128, sample_interval=50):
        
        # Configure input
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = X_train.reshape(X_train.shape[0],28,28)
        X_train = np.expand_dims(X_train, axis=3)
        y_train = y_train.reshape(-1, 1)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        lenet = build_lenet()
        lenet.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])
        
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, labels])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            X = np.concatenate((imgs.reshape(32,1,28,28), gen_imgs.reshape(32,1,28,28)))
            aux_y = np.concatenate((one_hot(np.squeeze(labels), 10), one_hot(np.squeeze(labels), 10)), axis=0)                
            epoch_lenet_loss = lenet.train_on_batch(X, aux_y)
            d_loss[0] = d_loss[0] + epoch_lenet_loss[0]

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            if epoch % sample_interval == 0:
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % 100 == 0:
                print(epoch)
                self.sample_images(epoch)
                
    def save_data(self,X_train, y_train, epochs, data_size):
        
        # Configure input
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        y_train = y_train.reshape(-1, 1)
        idx = np.random.randint(0, X_train.shape[0], data_size)
        labels = y_train[idx]
    
        # Sample noise as generator input
        noise = np.random.normal(0, 1, (data_size, 100))
    
        # Generate a half batch of new images
        gen_imgs = self.generator.predict([noise, labels])
        np.save("/content/Generate_data_%d_%d.npy"%(data_size,epochs),gen_imgs)
        np.save("/content/Generator_label_%d_%d.npy"%(data_size,epochs),labels)
    
    def sample_images(self, epoch):
        
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.arange(0, 10).reshape(-1, 1)
        gen_imgs = self.generator.predict([noise, sampled_labels])
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
                axs[i,j].set_title("Digit: %d" % sampled_labels[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("/content/drive/MyDrive/DE-Poise Capstone2/Images/%d.png" % epoch)
        plt.close()


def save_train_data(know_rate):

    path = '/content/drive/MyDrive/DE-Poise Capstone2/De-Pois-Code/mnist (5).npz'
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    know_number = int(x_train.shape[0] * know_rate)
    generator_size = int(x_train.shape[0] - know_number)
    data = x_train[0:know_number,:,:]
    label = y_train[0:know_number]
    np.save("train_data_%d.npy"%know_number,data)
    np.save("train_label_%d.npy"%know_number,label)
    
    return know_number, generator_size


def CGAN_data_loss(know_rate, epochs):
  
    know_rate = 0.2
    epochs = 2000
    know_number, generator_size = save_train_data(know_rate)
    train_data = np.load("train_data_%d.npy"%know_number)
    train_label = np.load("train_label_%d.npy"%know_number)

    start =time.perf_counter()
    cgan = CGAN()
    cgan.train(train_data, train_label,epochs, batch_size=32, sample_interval=10)
    cgan.save_data(train_data, train_label,epochs,generator_size)
    end = time.perf_counter()
    print('CGAN_data_loss Running time: %s Seconds'%(end-start))
    