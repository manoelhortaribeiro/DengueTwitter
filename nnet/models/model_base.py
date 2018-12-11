import tensorflow as tf
from keras.models import Model
from keras.callbacks import (ModelCheckpoint,
                             TensorBoard, ReduceLROnPlateau,
                             CSVLogger, EarlyStopping)
from keras.callbacks import TensorBoard
import keras.metrics as metrics
from keras.backend.tensorflow_backend import set_session
import keras_metrics

class ModelBase(object):

    def __init__(self, input_size, alphabet_size, embedding_size,
                 conv_layers, fully_connected_layers, num_of_classes,
                 threshold, dropout_p,
                 optimizer='adam', loss='categorical_crossentropy'):

        self.input_size = input_size
        self.alphabet_size = alphabet_size
        self.embedding_size = embedding_size
        self.conv_layers = conv_layers
        self.fully_connected_layers = fully_connected_layers
        self.num_of_classes = num_of_classes
        self.dropout_p = dropout_p
        self.optimizer = optimizer
        self.loss = loss
        self.threshold = threshold
        self._build_model()  # builds self.model variable

    def _build_model(self):
        pass

    def compile_model(self, inputs, predictions):
        # Build and compile model
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[metrics.categorical_accuracy])
        self.model = model
        self.model.summary()

    def train(self, training_inputs, training_labels,
              validation_inputs, validation_labels,
              epochs, batch_size):
        """
        Training function

        Args:
            training_inputs (numpy.ndarray): Training set inputs
            training_labels (numpy.ndarray): Training set labels
            validation_inputs (numpy.ndarray): Validation set inputs
            validation_labels (numpy.ndarray): Validation set labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            checkpoint_every (int): Interval for logging to Tensorboard

        Returns: None

        """

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        # Create callbacks
        # Create log
        callbacks = [ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       patience=7,
                                       min_lr=0.001),
                     EarlyStopping(patience=9,  # Patience should be larger than the one in ReduceLROnPlateau
                                   min_delta=0.0001)]
        callbacks += [TensorBoard(log_dir='./logs', batch_size=batch_size, write_graph=False),
                      CSVLogger('training.log', append=False)]  # Change append to true if continuing training
        # Save the BEST and LAST model
        callbacks += [ModelCheckpoint('./backup_model_last.hdf5'),
                      ModelCheckpoint('./backup_model_best.hdf5', save_best_only=True)]

        # Start training
        print("Training model: ")

        self.model.fit(training_inputs, training_labels,
                       validation_data=(validation_inputs, validation_labels),
                       epochs=epochs,
                       batch_size=batch_size,
                       verbose=2,
                       callbacks=callbacks)
    def test(self, testing_inputs, testing_labels, batch_size):
        """
        Testing function

        Args:
            testing_inputs (numpy.ndarray): Testing set inputs
            testing_labels (numpy.ndarray): Testing set labels
            batch_size (int): Batch size

        Returns: None

        """
        # Evaluate inputs
        self.model.evaluate(testing_inputs, testing_labels, batch_size=batch_size, verbose=1)
        # self.model.predict(testing_inputs, batch_size=batch_size, verbose=1)