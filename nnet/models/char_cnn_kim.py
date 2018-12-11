from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from keras.layers import Convolution1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import Embedding
from keras.layers import AlphaDropout
from .model_base import ModelBase


class CharCNNKim(ModelBase):
    """
    Class to implement the Character Level Convolutional Neural Network
    as described in Kim et al., 2015 (https://arxiv.org/abs/1508.06615)

    Their model has been adapted to perform text classification instead of language modelling
    by replacing subsequent recurrent layers with dense layer(s) to perform softmax over classes.
    """

    def _build_model(self):

        # Input layer
        inputs = Input(shape=(self.input_size,), name='sent_input', dtype='int64')
        # Embedding layers
        x = Embedding(self.alphabet_size + 1, self.embedding_size, input_length=self.input_size)(inputs)
        # Convolution layers
        convolution_output = []
        for num_filters, filter_width in self.conv_layers:
            conv = Convolution1D(filters=num_filters,
                                 kernel_size=filter_width,
                                 activation='tanh',
                                 name='Conv1D_{}_{}'.format(num_filters, filter_width))(x)
            pool = GlobalMaxPooling1D(name='MaxPoolingOverTime_{}_{}'.format(num_filters, filter_width))(conv)
            convolution_output.append(pool)
        x = Concatenate()(convolution_output)
        # Fully connected layers
        for fl in self.fully_connected_layers:
            x = Dense(fl, activation='selu', kernel_initializer='lecun_normal')(x)
            x = AlphaDropout(self.dropout_p)(x)
        # Output layer
        predictions = Dense(self.num_of_classes, activation='softmax')(x)

        self.compile_model(inputs, predictions)


