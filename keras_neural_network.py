from __future__ import absolute_import, division, print_function, unicode_literals
from keras.models import Sequential, load_model, save_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.activations import relu, tanh
from keras.initializers import random_normal


class KerasNN(object):

    def __init__(self,
                 input_neurons=64,
                 output_neurons=1,
                 neurons_by_hidden_layer=[64, 16, 6, 2, 1],
                 # neurons_by_hidden_layer=[64, 64, 64, 5, 1],
                 act_func=[tanh, tanh, tanh, tanh, tanh, tanh],
                 kernels=[random_normal(stddev=0.1, seed=101),
                          random_normal(stddev=0.1, seed=101),
                          random_normal(stddev=0.1, seed=101),
                          random_normal(stddev=0.1, seed=101),
                          random_normal(stddev=0.1, seed=101),
                          random_normal(stddev=0.1, seed=101)],
                 biases=['ones', 'ones', 'ones', 'ones', 'ones', 'ones'],
                 loss='mse',
                 optimizer='adam',
                 metrics=[],
                 verbose=0,
                 model_file=None):
        """Constructor."""
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.initial_epoch = 0
        self.epochs = 0
        self.hidden_layers = neurons_by_hidden_layer
        self.verbose = verbose
        self.model_file = model_file  # Model file path
        self.model = self.initialize_model(kernels, biases, act_func)
        self.compile()

    def initialize_model(self, kernels, biases, act_fn):
        """Initialize network"""
        model = Sequential()
        if len(kernels) == 1:
            model.add(Dense(self.output_neurons,
                            input_dim=self.input_neurons,
                            kernel_initializer=kernels[0],
                            bias_initializer=biases[0],
                            activation=act_fn[0]))
        else:
            for i in range(len(kernels)):
                if i == 0:
                    model.add(Dense(self.hidden_layers[i],
                                    input_dim=self.input_neurons,
                                    kernel_initializer=kernels[i],
                                    bias_initializer=biases[i],
                                    activation=act_fn[i]))
                elif i == (len(kernels) - 1):
                    model.add(Dense(self.output_neurons,
                                    kernel_initializer=kernels[i],
                                    bias_initializer=biases[i],
                                    activation=act_fn[i]))
                else:
                    model.add(Dense(self.hidden_layers[i],
                                    kernel_initializer=kernels[i],
                                    bias_initializer=biases[i],
                                    activation=act_fn[i]))

        # rospy.loginfo("KerasNN model initialized.")
        return model

    def compile(self):
        """Configures the model for training"""
        # try:
        if self.model_file is not None:
            self.model = load_model(self.model_file)
            # rospy.loginfo("Compiling saved Keras model in " + str(self.model_file))
        else:
            self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
            # rospy.loginfo("Compiling Keras model: optimizer -> " + str(self.optimizer) + ", loss function -> " + str(self.loss))
        # except ValueError as e:
            # rospy.logerr("Error when compiling model... " + str(e))

    def save_model(self, filepath):
        """Save a Keras model into a single HDF5 file which will contain:
            - the architecture of the model, allowing to re-create the model
            - the weights of the model
            - the training configuration (loss, optimizer)
            - the state of the optimizer, allowing to resume training exactly where you left off"""
        # try:
        save_model(self.model, str(filepath), include_optimizer=True)
            # rospy.loginfo("Saving Keras model in " + str(filepath))
        # except ImportError as e:
        #     rospy.logerr("Error when saving Keras model..." + str(e))

    def load_model(self, filepath):
        """Reinstantiate a Keras model from a HDF5 file.
         load_model will also take care of compiling the model using the saved training 
         configuration (unless the model was never compiled in the first place)."""
        # try:
        self.model = load_model(str(filepath))
            # rospy.loginfo("Loading Keras model from " + str(filepath))
        # except ImportError as e:
        #     rospy.logerr("Error when loading Keras model..." + str(e))
        # except ValueError as e:
        #     rospy.logerr("Error when loading Keras model..." + str(e))

    def train(self, input_data, output_data, batch_size=None, epochs=10, validation_data=None, validation_split=0.1):
        """Trains the model for a given number of epochs (iterations on a dataset)."""
        history = None
        self.epochs += epochs
        # simple early stopping
        es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
        # try:
            # rospy.loginfo("Training Keras model...")
        history = self.model.fit(
            x=input_data,
            y=output_data,
            batch_size=batch_size,
            epochs=self.epochs,
            verbose=self.verbose,
            callbacks=[es],
            initial_epoch=self.initial_epoch,
            validation_data=validation_data,  # Tuple (x_val, y_val). Validation_data will override validation_split.

            validation_split=validation_split  # Fraction of the training data to be used as validation data.
        )
        self.initial_epoch += epochs
        # except ValueError as e:
        #     rospy.logerr("Error in training..." + str(e))
        # except RuntimeError as _:
        #     rospy.logerr("Error in training...the model was never compiled.")
        return history

    def predict(self, input_data):
        """Generates output predictions for the input samples."""
        predictions = None
        # try:
            # rospy.loginfo("Obtaining predictions...")
        predictions = self.model.predict(input_data, verbose=self.verbose)
        # except ValueError as e:
        #     rospy.logerr("Error when predicting..." + str(e))
        return predictions

    def evaluate(self, input_data, output_data, batch_size=None):
        """Returns the loss value & metrics values for the model in test mode.
        Computation is done in batches."""
        loss = None
        # try:
            # rospy.loginfo("Evaluating Keras model...")
        loss = self.model.evaluate(
            x=input_data,
            y=output_data,
            batch_size=batch_size,
            verbose=self.verbose
        )
        # except ValueError as e:
        #     rospy.logerr("Error when evaluating model..." + str(e))
        # except RuntimeError as e:
        #     rospy.logerr("Error when evaluating model..." + str(e))
        return loss

    def get_model(self):
        return self.model
