from tensorflow.keras.layers import Activation, UpSampling2D, Reshape, Input, Conv2D, \
    LeakyReLU, BatchNormalization, Flatten, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import librosa.display
import numpy as np
import os
import cv2
import pandas as pd
import scipy.io.wavfile
import time


MODEL_DIR = "./saved_models/"
STFT_DIR = "./data/resized_stft/"
AUDIO_OUT_DIR = "./output/audios/"
STFT_OUT_DIR = "./output/arrays/"
sampling_rate = 22500
eps = 1e-7
beta = 0.9

df = pd.read_csv("./data/saved_mean_std.csv")
means = df['mean']
stds = df['std']
mag_mean = 0
for index in range(means.shape[0]):
    if means[index] != -np.inf:
        mag_mean += means[index] / means.shape[0]
mag_std = np.mean(stds)
losses = {"epoch": [], "d_loss": [], "d_acc": [], "g_loss": []}


# gan overall structure from CS230 lecture slides by Kian Katanforoosh
class pianoGAN():
    def __init__(self):
        self.height = 256
        self.width = 256
        self.latent_dim = 100
        self.shape = (self.height, self.width, 1)

        # https://machinelearningmastery.com/how-to-train-stable-generative-adversarial-networks/
        optimizer = Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9, clipnorm=1)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        self.generator = self.build_generator()

        z = Input(shape=(self.latent_dim,))
        matrix = self.generator(z)
        self.discriminator.trainable = False
        validity = self.discriminator(matrix)
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(256 * 256, input_dim=self.latent_dim))
        model.add(Reshape((16, 16, 256)))  # 16 * 16
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D(size=(2, 2)))  # 32 * 32
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D(size=(2, 2)))  # 64 * 64
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D(size=(2, 2)))  # 128 * 128
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(16, kernel_size=3, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D(size=(2, 2)))  # 256 * 256
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
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        matrix = Input(shape=self.shape)
        validity = model(matrix)

        return Model(matrix, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):
        t = time.time()
        paths = self.get_dataset_paths(STFT_DIR, ".npy")
        all_arrays = []
        for path in paths:
            all_arrays.append(np.load(path))
        S_train = np.expand_dims(np.stack(all_arrays), axis=-1)

        print("All training data loaded, took", time.time() - t, "seconds.")

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            t = time.time()
            idx = np.random.randint(0, S_train.shape[0], batch_size)
            matrices = S_train[idx]
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            gen_matrices = self.generator.predict(noise)
            d_loss_real = self.discriminator.train_on_batch(matrices, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_matrices, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)

            print("Finished epoch", epoch, "after", time.time() - t, "seconds")
            print("Discriminator loss and accuracy:", d_loss, "Generator loss:", g_loss)

            if epoch % sample_interval == 0:
                self.generator.save(MODEL_DIR + "generator")
                self.discriminator.save(MODEL_DIR + "discriminator")
                out = AUDIO_OUT_DIR + "epoch_" + str(epoch) + ".wav"
                try:
                    y = self.audio_reconstruction(gen_matrices[0])
                    scipy.io.wavfile.write(out, sampling_rate, y)
                except:
                    print("Error in griffinlim at epoch", epoch)
                    pass
                out = STFT_OUT_DIR + "epoch_" + str(epoch) + ".npy"
                np.save(out, gen_matrices[0])
                losses["d_loss"].append(d_loss[0])
                losses["d_acc"].append(d_loss[1])
                losses["g_loss"].append(g_loss)
                losses["epoch"].append(epoch)
                df = pd.DataFrame(losses, columns=['epoch', 'd_loss', 'd_acc', 'g_loss'])
                df.to_pickle("./output/loss.pkl")

    def get_dataset_paths(self, directory, extension):
        paths = []
        for subdir, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(extension):
                    path = os.path.join(subdir, file)
                    paths.append(path)
        return paths

    def upsample(self, S_downsample):
        return cv2.resize(S_downsample, (862, 1025), interpolation=cv2.INTER_LINEAR)

    def audio_reconstruction(self, S):
        S = self.upsample(np.squeeze(S))
        S = S * 3
        S = S * (mag_std + eps) + mag_mean
        S = np.exp(S)

        return librosa.griffinlim(S)


if __name__ == '__main__':
    gan = pianoGAN()
    gan.train(epochs=15000, batch_size=128, sample_interval=100)
    gan.generator.save(MODEL_DIR + "generator")
    gan.discriminator.save(MODEL_DIR + "discriminator")
    df = pd.read_pickle("./output/loss.pkl")
    print(df)
