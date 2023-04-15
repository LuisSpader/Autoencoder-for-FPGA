import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from callbacks import all_callbacks
from matplotlib import pyplot as plt
import MNIST_database as mnist


class QAutoencoder:
    def __init__(self, data: mnist.MNISTData, bit_width=8, model_name='model'):

        self.x_train = data.x_train
        self.y_train = data.y_train
        self.x_test = data.x_test
        self.y_test = data.y_test
        self.input_shape = (data.x_train.shape[-1],)
        self.BIT_WIDTH = bit_width
        self.MODEL_NAME = model_name
        self.model = self.autoencoder_generator()
        self.history = None
        self.loss = None

        self.interpreter = None
        self.quantized_tflite_model = None
        self.input_details = None
        self.output_details = None

    def autoencoder_generator(self):

        # Encoder
        encoder_input = Input(shape=self.input_shape)
        encoder_l1 = Dense(64, activation='relu')(encoder_input)
        encoder_l2 = Dense(32, activation='relu')(encoder_l1)
        encoder_l3 = Dense(16, activation='relu')(encoder_l2)
        encoder_output = Dense(2, activation='relu')(encoder_l3)

        # Decoder
        decoder_l1 = Dense(16, activation='relu')(encoder_output)
        decoder_l2 = Dense(32, activation='relu')(decoder_l1)
        decoder_l3 = Dense(32, activation='relu')(decoder_l2)
        # decoder_output = Dense(y_train.shape[-1], activation='sigmoid')(decoder_l3) # classifier
        decoder_output = Dense(
            self.x_train.shape[-1], activation='sigmoid')(decoder_l3)  # autoencoder

        # Model
        model = Model(inputs=encoder_input, outputs=decoder_output)
        # refactor the code above to use the functional AP

        model.compile(optimizer='adam', loss='mse')
        # model.compile(optimizer='adam', loss='binary_crossentropy') # classifier
        return model

    def fit_data(self, batch_size=256, epochs=30):
        """Write the fit function for the autoencoder. 
        Storing the fit history in self.history to be able to plot the fitting scores."""

        callbacks = all_callbacks(stop_patience=1000,
                                  lr_factor=0.5,
                                  lr_patience=10,
                                  lr_epsilon=0.000001,
                                  # min_delta=0.000001,
                                  lr_cooldown=2,
                                  lr_minimum=0.0000001,
                                  outputDir=f'model/QAE_model{self.BIT_WIDTH}bits/callbacks')
        # callbacks.callbacks.append(pruning_callbacks.UpdatePruningStep())

        self.history = self.model.fit(self.x_train, self.x_train,
                                      validation_data=(
                                          self.x_test, self.x_test),
                                      batch_size=batch_size, epochs=epochs,
                                      shuffle=True, callbacks=callbacks.callbacks)
        # self.model = strip_pruning(self.model)
        self.model.save(
            # f'model/QAE_model{self.BIT_WIDTH}bits/KERAS_check_best_model.h5')
            f'model/QAE_model{self.BIT_WIDTH}bits/KERAS_check_best_model.model')
        self.history = self.history.history
        self.loss = self.model.evaluate(self.x_test, self.x_test, verbose=0)
        self.convert_to_tflite()

    def plot_float_model(self, n=6):
        """Plot the float model"""
        # test_imgs = x_train
        test_imgs = self.x_test

        plt.figure(figsize=(10, 3))
        quantized_model_predictions = self.model.predict(test_imgs)
        self._extracted_from_plot_quantized_model_8(
            n,
            test_imgs,
            quantized_model_predictions,
            './images/QAE/reconstructed images {model_name}.png',
        )

    def representative_dataset(self):
        for data in self.x_train:
            # yield [np.array([data], dtype=np.float32)]
            yield [np.array([data * (2 ** (self.BIT_WIDTH - 1))], dtype=np.float32)]

    def convert_to_tflite(self):

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = self.representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        self.quantized_tflite_model = converter.convert()
        # Save the quantized model
        with open(f'quantized_model.tflite', 'wb') as f:
            f.write(self.quantized_tflite_model)

        # Load the quantized model
        self.interpreter = tf.lite.Interpreter(
            model_path='quantized_model.tflite')
        self.interpreter.allocate_tensors()

        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def plot_quantized_model(self, n=6):
        quantized_model_predictions = []
        # test_imgs = x_train
        test_imgs = self.x_test

        for i in range(n):
            # Prepare input data
            input_data = np.array(
                [test_imgs[i]*(2**(self.BIT_WIDTH-1))], dtype=np.int8)
            self.interpreter.set_tensor(
                self.input_details[0]['index'], input_data)

            # Run inference
            self.interpreter.invoke()

            # Get output
            # output_data = interpreter.get_tensor(output_details[0]['index'])
            output_data = self.interpreter.get_tensor(
                self.output_details[0]['index']) / (2 ** (self.BIT_WIDTH - 1))
            quantized_model_predictions.append(output_data)

        plt.figure(figsize=(10, 3))
        self._extracted_from_plot_quantized_model_8(
            n,
            test_imgs,
            quantized_model_predictions,
            './images/QAE/reconstructed images{model_name}.png',
        )

    # TODO Rename this here and in `plot_float_model` and `plot_quantized_model`
    def _extracted_from_plot_quantized_model_8(self, n, test_imgs, quantized_model_predictions, arg3):
        img_size = int(np.sqrt(self.input_shape[0]))
        for i in range(n):
            ax = plt.subplot(2, n, i + 1)
            self.plot_imgs(
                test_imgs, i, img_size, ax
            )
            ax = plt.subplot(2, n, i + n + 1)
            self.plot_imgs(
                quantized_model_predictions, i, img_size, ax
            )
        plt.savefig(arg3.format(model_name=" complete"))
        plt.show()

    # TODO Rename this here and in `plot_float_model` and `plot_quantized_model`
    def plot_imgs(self, arg0, i, img_size, ax):
        plt.imshow(arg0[i].reshape(img_size, img_size), cmap='gray_r')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
