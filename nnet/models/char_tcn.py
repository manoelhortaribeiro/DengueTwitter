from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import Convolution1D
from keras.layers import Embedding
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import SpatialDropout1D
from keras.layers import Dropout
from keras.layers import Add
import keras.metrics as metrics
from .model_base import ModelBase


class CharTCN(ModelBase):
    """
    Class to implement the Character Level Temporal Convolutional Network (TCN)
    as described in Bai et al., 2018 (https://arxiv.org/pdf/1803.01271.pdf)
    """

    def _build_model(self):
        # Input layer
        inputs = Input(shape=(self.input_size,), name='sent_input', dtype='int64')
        # Embedding layers
        x = Embedding(self.alphabet_size + 1, self.embedding_size, input_length=self.input_size)(inputs)
        # Residual blocks with 2 Convolution layers each
        d = 1  # Initial dilation factor
        for cl in self.conv_layers:
            res_in = x
            for _ in range(2):
                # NOTE: The paper used padding='causal'
                x = Convolution1D(cl[0], cl[1], padding='same', dilation_rate=d, activation='linear')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = SpatialDropout1D(self.dropout_p)(x)
                d *= 2  # Update dilation factor
            # Residual connection
            res_in = Convolution1D(filters=cl[0], kernel_size=1, padding='same', activation='linear')(res_in)
            x = Add()([res_in, x])
        x = Flatten()(x)
        # Fully connected layers
        for fl in self.fully_connected_layers:
            x = Dense(fl)(x)
            x = Activation('relu')(x)
            x = Dropout(self.dropout_p)(x)
        # Output layer
        predictions = Dense(self.num_of_classes, activation='softmax')(x)

        self.compile_model(inputs, predictions)

