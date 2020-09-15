import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import losses
from sklearn.model_selection import train_test_split

class AutoEnconder(object):

    def __init__(self, epochs=10, batch_size=32, validation_size=0.2,
                 random_state=42, regularizer = [],
                 exposure = None, modality = None):

        # auto-enconder parameters
        self.value_encoding_dim = []
        self.regularizer = regularizer
        self.batch_size = batch_size
        self.epochs = epochs
        
        # validation split parameters
        self.test_size = validation_size
        self.random_state = random_state
        
        # information about when is use
        self.exposure = exposure
        self.modality = modality

        # information about history train
        self.history = []

        # saving method
        self.method_autoenconder = []
        self.method_enconder = []

    def make_auto_enconder(self, value_encoding_dim):

        self.value_encoding_dim = value_encoding_dim
        
        fun_loss = losses.mean_absolute_percentage_error

        original_trial = Input(shape=(62))
        enconded = Dense(62, activation='relu')(original_trial)
        enconded = Dense(32, activation='relu')(enconded)
        enconded = Dense(16, activation='relu')(enconded)
        enconded = Dense(8, activation='relu')(enconded)
        if (self.regularizer != []):
            enconded = Dense(self.value_encoding_dim,
                             activation='relu',
                             activity_regularizer=self.regularizer
                             )(enconded)
        else:
            enconded = Dense(self.value_encoding_dim,
                             activation='relu')(enconded)

        deconded = Dense(8, activation='relu', use_bias=False)(enconded)
        deconded = Dense(16, activation='relu', use_bias=False)(deconded)
        deconded = Dense(32, activation='relu', use_bias=False)(deconded)
        deconded = Dense(62, activation='relu', use_bias=False)(deconded)

        encoder = Model(original_trial, enconded, name='encoder')
        autoencoder = Model(original_trial, deconded, name='autoenconder')

        autoencoder.compile(optimizer='adam', loss=fun_loss,
                            metrics=['accuracy'])

        self.method_autoenconder = autoencoder
        self.method_enconder = encoder

    def fit(self, X):

        #import pdb; pdb.set_trace()
        X_train = X.T.to_numpy()
        X_validation = X_train.copy()

        # Training auto-enconder
        self.history = self.method_autoenconder.fit(X_train, X_train,
                                       epochs=self.epochs,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       validation_data=(
                                           X_validation, X_validation),
                                       verbose=0)
    def transform(self, X):
        
        return self.method_enconder.predict(X)