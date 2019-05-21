import os
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard
from base.base_trainer import BaseTrain
from trainers.wikiart_data_generator import WikiArtDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from utils.utils import calc_degrees

class ResNet50ModelTrainer(BaseTrain):
    def __init__(self, model, data, val_data, config, tensorboard_log_dir, checkpoint_dir):
        super(ResNet50ModelTrainer, self).__init__(model, data, config)
        self.tensorboard_log_dir = tensorboard_log_dir
        self.checkpoint_dir = checkpoint_dir   
        self.val_data = val_data
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
            experiment = Experiment(api_key=self.config['comet_api_key'], project_name=self.config['exp_name'])
            experiment.disable_mp()
            experiment.log_multiple_params(self.config)
            self.callbacks.append(experiment.get_keras_callback())      

    def train(self):
        train_datagen = WikiArtDataGenerator(img_filenames=self.data[0], labels=self.data[1], batch_size=self.config['batch_size'], target_size=(self.config['img_size'], self.config['img_size']))
        val_datagen = WikiArtDataGenerator(img_filenames=self.val_data[0], labels=self.val_data[1], batch_size=self.config['batch_size'], target_size=(self.config['img_size'], self.config['img_size']))

        history = self.model.fit_generator(
            train_datagen,
            steps_per_epoch = self.data[0].shape[0]/self.config['batch_size'],
            epochs = self.config['nb_epoch'],
            initial_epoch = self.config['initial_epoch'],
            verbose = self.config['verbose_training'],
            validation_data = val_datagen,
            validation_steps = 1,
            # TODO: Split training data into validation
            # validation_split=self.config['validation_split'],
            # TODO: Add LR Scheduler
            # lr_scheduler,
            callbacks = self.callbacks,
        )
        self.loss.extend(history.history['loss'])
        self.val_loss.extend(history.history['val_loss'])

    def denormalize_attr(self, val):
        min_val = 1
        max_val = 10
        return (((val + 1)/2) * (max_val - min_val)) + min_val

    def predict(self):
        # turn off shuffling
        val_datagen = WikiArtDataGenerator(img_filenames=self.val_data[0], labels=self.val_data[1], batch_size=self.config['batch_size'], target_size=(self.config['img_size'], self.config['img_size']), shuffle=False)

        # predict
        Y_pred = self.model.predict_generator(val_datagen, len(val_datagen))

        # print predicted values
        for i in range(len(Y_pred[0])):
            # print filename and values
            print(self.val_data[0][i])

            len_numerical_attributes = len(self.config['numerical_loss_weights'])
            for j, attr in enumerate(self.config['numerical_loss_weights'].keys()):
                print('{0}: {1} pred: {2:0.1f}'.format(attr, self.denormalize_attr(self.val_data[1][i][j]), self.denormalize_attr(Y_pred[j][i][0])))
            
            # HSV color
            print(list(self.config['categorical_loss_weights'])[0])

            # Actual HSV
            print(self.val_data[1][i][len_numerical_attributes])         
            print('actual degrees: {0:0.0f}'.format(calc_degrees(self.val_data[1][i][len_numerical_attributes][0], self.val_data[1][i][len_numerical_attributes][1])))

            # Predicted HSV
            print(Y_pred[len_numerical_attributes][i])
            print('pred degrees: {0:0.1f}'.format(calc_degrees(Y_pred[len_numerical_attributes][i][0], Y_pred[len_numerical_attributes][i][1])))


            print('{0}: {1} pred: {2}\n'.format(list(self.config['categorical_loss_weights'])[1], self.config['harmonies'][np.argmax(self.val_data[1][i][1 + len_numerical_attributes])], self.config['harmonies'][np.argmax(Y_pred[1 + len_numerical_attributes][i])]))


        # TODO: pass labels instead of values to confusion matrices
        # labels_onehot = np.array(self.val_data[1], dtype=object)

        # color_pred = np.argmax(Y_pred[6], axis=1)
        # color_labels = np.argmax(labels_onehot[:,6].tolist(), axis=1)
        # print('Color Confusion Matrix')
        # print(confusion_matrix(color_labels, color_pred, labels=self.config['colors']))
        # print('Classification Report')
        # print(classification_report(color_labels, color_pred, target_names=self.config['colors']))

        # harmony_pred = np.argmax(Y_pred[7], axis=1)
        # harmony_labels = np.argmax(labels_onehot[:,7].tolist(), axis=1)
        # print('Harmony Confusion Matrix')
        # print(confusion_matrix(harmony_labels, harmony_pred, labels=self.config['harmonies']))
        # print('Classification Report')
        # print(classification_report(harmony_labels, harmony_pred, target_names=self.config['harmonies']))