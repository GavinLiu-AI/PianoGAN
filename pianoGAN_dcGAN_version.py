from tensorflow.keras.layers import Activation, UpSampling2D, Reshape, Input, Conv2D, \
    LeakyReLU, BatchNormalization, Flatten, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import scipy.io.wavfile
import time
import util

MODEL_DIR = "./saved_models/"
STFT_DIR = "./data/stft_arrays/"
AUDIO_OUT_DIR = "./output/audios/"
STFT_OUT_DIR = "./output/arrays/"
sampling_rate = 10000
eps = 1e-7
beta = 0.9

losses = {"d_loss": [], "d_acc": [], "g_loss": []}


'''
Based on and modified from: 
    Title: Keras-GAN
    Author: Erik Linder-Norén
    Date: 2017
    Code version: -
    Availability: https://github.com/eriklindernoren/Keras-GAN.git
'''
class pianoGAN():
    def __init__(self):

        # define input size
        self.height = 512
        self.width = 512
        self.latent_dim = 100
        self.shape = (self.height, self.width, 1)

        # define optimizer
        optimizer = Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9, clipnorm=1)

        # build discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
        # build generator
        self.generator = self.build_generator()

        # construct noisy data
        z = Input(shape=(self.latent_dim,))
        matrix = self.generator(z)

        # we do not train the discriminator
        self.discriminator.trainable = False

        # test validity of input
        validity = self.discriminator(matrix)
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(512 * 512, input_dim=self.latent_dim))
        model.add(Reshape((32, 32, 256)))  # 32 * 32
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D(size=(2, 2)))  # 64 * 64
        model.add(Conv2D(512, kernel_size=3, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D(size=(2, 2)))  # 128 * 128
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D(size=(2, 2)))  # 256 * 256
        model.add(Conv2D(32, kernel_size=3, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(8, kernel_size=3, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D(size=(2, 2)))  # 512 * 512
        model.add(Conv2D(1, kernel_size=3, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Activation("tanh"))

        model.add(Reshape(self.shape))
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        matrix = model(noise)

        return Model(noise, matrix)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, input_shape=self.shape, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, input_shape=self.shape, kernel_size=3, stride=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(512, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        matrix = Input(shape=self.shape)
        validity = model(matrix)

        return Model(matrix, validity)

    def train(self, epochs, batch_size=128, sample_interval=1):
        t = time.time()

        # acquire training data
        paths = util.get_dataset_paths(STFT_DIR, ".npy")
        all_arrays = []
        for index in range(len(paths[0: 10000])):
            all_arrays.append(np.load(paths[index]))
            if index % 5000 == 0 and index != 0:
                print(index, "training data loaded.")
        S_train = np.expand_dims(np.stack(all_arrays), axis=-1)

        print("All training data loaded, size", S_train.shape[0], ", took", time.time() - t, "seconds.")

        # train each epoch
        for epoch in range(epochs):
            t = time.time()
            print("Running epoch", epoch, "...")
            np.random.shuffle(S_train)
            gen_matrix = None

            # run each batch training
            for iteration in range(int(S_train.shape[0] / batch_size)):

                # define true label and fake label
                # insert noise by flipping part of the labels
                valid = np.ones((batch_size, 1))
                fake = np.zeros((batch_size, 1))
                valid[0: max(0, 8 - epoch)] = 0
                fake[0: max(0, 8 - epoch)] = 1
                np.random.shuffle(valid)
                np.random.shuffle(fake)

                # obtain training batch
                S_train_batch = S_train[iteration * batch_size: (iteration + 1) * batch_size]

                # input for generator
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # construct noisy input to weaken discriminator
                noise_dis = np.random.normal(0, 1, (max(int((batch_size - epoch) / 2), 0),
                                                     S_train.shape[1], S_train.shape[2], S_train.shape[3]))
                noise_dis = np.where(np.abs(noise_dis) < 1, noise_dis, np.sign(noise_dis))
                noise_label = np.zeros((max(batch_size - epoch, 0), 1))
                noise_label[0: int(batch_size / 2)] = 1
                np.random.shuffle(noise_label)

                # run generator and discriminator, record loss and accuracy
                gen_matrices = self.generator.predict(noise)
                d_loss_real = self.discriminator.train_on_batch(np.concatenate((S_train_batch, noise_dis), axis=0),
                                                                np.concatenate((valid, noise_label), axis=0))
                d_loss_fake = self.discriminator.train_on_batch(np.concatenate((gen_matrices, noise_dis), axis=0),
                                                                np.concatenate((fake, noise_label), axis=0))
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                valid = np.ones((batch_size, 1))
                g_loss = self.combined.train_on_batch(noise, valid)
                losses["d_loss"].append(d_loss[0])
                losses["d_acc"].append(d_loss[1])
                losses["g_loss"].append(g_loss)
                gen_matrix = gen_matrices[0]

                print("In epoch", epoch, "at iteration", iteration)
                print("Discriminator loss and accuracy:", d_loss, "Generator loss:", g_loss)

            print("Epoch", epoch, "took", time.time() - t, "seconds.")

            # if current epoch is a saving epoch
            if epoch % sample_interval == 0:

                # save the generator and discriminator
                self.generator.save(MODEL_DIR + "generator")
                self.discriminator.save(MODEL_DIR + "discriminator")

                # revert generated stft to wav file and save to output folder
                out = AUDIO_OUT_DIR + "epoch_" + str(epoch) + ".wav"
                try:
                    y = util.audio_reconstruction(gen_matrix)
                    scipy.io.wavfile.write(out, sampling_rate, y)
                except:
                    print("Error in griffinlim at epoch", epoch)
                    pass

                # save generated matrix to output folder
                out = STFT_OUT_DIR + "epoch_" + str(epoch) + ".npy"
                np.save(out, gen_matrix)

                # record loss and accuracy
                df = pd.DataFrame(losses, columns=['d_loss', 'd_acc', 'g_loss'])
                df.to_pickle("./output/loss.pkl")


if __name__ == '__main__':
    gan = pianoGAN()
    gan.train(epochs=100, batch_size=32, sample_interval=1)
    gan.generator.save(MODEL_DIR + "generator")
    gan.discriminator.save(MODEL_DIR + "discriminator")
    util.plot_loss(("./output/loss.pkl"))
