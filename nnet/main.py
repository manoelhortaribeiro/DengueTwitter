import tensorflow as tf
import json

from nnet.data_utils import Data
from nnet.models.char_cnn_zhang import CharCNNZhang
from nnet.models.char_cnn_kim import CharCNNKim
from nnet.models.char_tcn import CharTCN

model_v = "zhang"

if __name__ == "__main__":
    # Load configurations
    config = json.load(open("./nnet/config.json"))
    # Load training data
    training_data = Data(data_source=config["data"]["training_data_source"],
                         alphabet=config["data"]["alphabet"],
                         input_size=config["data"]["input_size"],
                         num_of_classes=config["data"]["num_of_classes"])
    training_data.load_data()
    training_inputs, training_labels = training_data.get_all_data()
    # Load validation data
    validation_data = Data(data_source=config["data"]["validation_data_source"],
                           alphabet=config["data"]["alphabet"],
                           input_size=config["data"]["input_size"],
                           num_of_classes=config["data"]["num_of_classes"])
    validation_data.load_data()
    validation_inputs, validation_labels = validation_data.get_all_data()

    # Load model configurations and build model
    if model_v == "kim":
        model = CharCNNKim(input_size=config["data"]["input_size"],
                           alphabet_size=config["data"]["alphabet_size"],
                           embedding_size=config["char_cnn_kim"]["embedding_size"],
                           conv_layers=config["char_cnn_kim"]["conv_layers"],
                           fully_connected_layers=config["char_cnn_kim"]["fully_connected_layers"],
                           num_of_classes=config["data"]["num_of_classes"],
                           dropout_p=config["char_cnn_kim"]["dropout_p"],
                           optimizer=config["char_cnn_kim"]["optimizer"],
                           loss=config["char_cnn_kim"]["loss"])
    elif model_v == 'tcn':
        model = CharTCN(input_size=config["data"]["input_size"],
                        alphabet_size=config["data"]["alphabet_size"],
                        embedding_size=config["char_tcn"]["embedding_size"],
                        conv_layers=config["char_tcn"]["conv_layers"],
                        fully_connected_layers=config["char_tcn"]["fully_connected_layers"],
                        num_of_classes=config["data"]["num_of_classes"],
                        dropout_p=config["char_tcn"]["dropout_p"],
                        optimizer=config["char_tcn"]["optimizer"],
                        loss=config["char_tcn"]["loss"])
    else:
        model = CharCNNZhang(input_size=config["data"]["input_size"],
                             alphabet_size=config["data"]["alphabet_size"],
                             embedding_size=config["char_cnn_zhang"]["embedding_size"],
                             conv_layers=config["char_cnn_zhang"]["conv_layers"],
                             fully_connected_layers=config["char_cnn_zhang"]["fully_connected_layers"],
                             num_of_classes=config["data"]["num_of_classes"],
                             threshold=config["char_cnn_zhang"]["threshold"],
                             dropout_p=config["char_cnn_zhang"]["dropout_p"],
                             optimizer=config["char_cnn_zhang"]["optimizer"],
                             loss=config["char_cnn_zhang"]["loss"])
    # Train model
    model.train(training_inputs=training_inputs,
                training_labels=training_labels,
                validation_inputs=validation_inputs,
                validation_labels=validation_labels,
                epochs=config["training"]["epochs"],
                batch_size=config["training"]["batch_size"])
