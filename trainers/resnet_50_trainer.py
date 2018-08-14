import os
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard
from base.base_trainer import BaseTrain

class ResNet50ModelTrainer(BaseTrain):
    def __init__(self, model, data, test_data, config, tensorboard_log_dir, checkpoint_dir):
        super(ResNet50ModelTrainer, self).__init__(model, data, config)
        self.tensorboard_log_dir = tensorboard_log_dir
        self.checkpoint_dir = checkpoint_dir   
        self.test_data = test_data
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath = os.path.join(self.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config['exp_name']),
                monitor = self.config['checkpoint_monitor'],
                mode = self.config['checkpoint_mode'],
                save_best_only = self.config['checkpoint_save_best_only'],
                save_weights_only = self.config['checkpoint_save_weights_only'],
                verbose = self.config['checkpoint_verbose'],
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir = self.tensorboard_log_dir,
                write_graph = self.config['tensorboard_write_graph'],
                histogram_freq = 0, # don't compute histograms
                write_images = False # don't write model weights to visualize as image in TensorBoard
            )
        )

        if hasattr(self.config,"comet_api_key"):
            from comet_ml import Experiment
            experiment = Experiment(api_key=self.config.comet_api_key, project_name=self.config.exp_name)
            experiment.disable_mp()
            experiment.log_multiple_params(self.config)
            self.callbacks.append(experiment.get_keras_callback())

    def batch_generator(self, x, y, batch_size):
        idx = 0
        while True:
            X_batch = x[idx:idx+batch_size]
            Y_batch = y[idx:idx+batch_size]

            # transpose y values to create list of numpy arrays by attribute
            Y_batch_transposed = list(map(list, zip(*Y_batch)))

            idx += batch_size

            if idx == x.shape[0]:
                idx = 0
            elif idx > x.shape[0] - batch_size:
                idx = x.shape[0] - batch_size

            yield X_batch, [np.array(Y_batch_transposed[0]), np.array(Y_batch_transposed[1]), np.array(Y_batch_transposed[2]),
                np.array(Y_batch_transposed[3]),
                np.array(Y_batch_transposed[4]), np.array(Y_batch_transposed[5]), np.array(Y_batch_transposed[6])]            

    def train(self):
        history = self.model.fit_generator(
            self.batch_generator(self.data[0], self.data[1], batch_size = self.config['batch_size']),
            steps_per_epoch = self.data[0].shape[0]/self.config['batch_size'],
            epochs = self.config['nb_epoch'],
            initial_epoch = self.config['initial_epoch'],
            verbose = self.config['verbose_training'],
            validation_data = (self.test_data[0], self.test_data[1]),
            # TODO: Split training data into validation
            # validation_split=self.config['validation_split'],
            # TODO: Add LR Scheduler
            # lr_scheduler,
            callbacks = self.callbacks,
        )
        self.loss.extend(history.history['loss'])
        self.val_loss.extend(history.history['val_loss'])