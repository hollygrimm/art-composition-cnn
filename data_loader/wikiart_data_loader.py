import os
import numpy as np
import pandas as pd
from keras.preprocessing import image
from base.base_data_loader import BaseDataLoader
from keras.applications.imagenet_utils import preprocess_input

class WikiArtDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(WikiArtDataLoader, self).__init__(config)

        # read only the necessary columns from the CSV
        usecols = ['in_train', 'new_filename']
        attributes = [k for k in config['loss_weights']]
        usecols.extend(attributes)
        df = pd.read_csv(config['train_test_csv_file'], usecols=usecols)

        train_data = []
        train_labels = []
        test_data = []
        test_labels = []    

        # train on only those rows that have attribute information
        filtered = df.dropna(subset=[attributes[0]])

        for in_train, new_filename, variety_texture, variety_shape, variety_size, variety_color, variety, contrast, repetition in filtered.values:
            attributes = {'variety_texture': variety_texture, 'variety_shape': variety_shape, 'variety_size': variety_size,
                'variety_color': variety_color,
                'variety': variety, 'contrast': contrast, 'repetition': repetition}

            # normalize attribute values input: [1, 10]   output: [-1, 1]
            min_val = 1
            max_val = 10
            norm_attrs = [(2 * ((v - min_val)/(max_val - min_val))) - 1 for k, v in attributes.items()]

            if in_train:
                img_file_path = os.path.join('data/train', new_filename)
                img_data = self.prepare_image(img_file_path, (config['img_size'], config['img_size']))
                train_data.append(img_data)
                train_labels.append(norm_attrs)
            else:
                img_file_path = os.path.join('data/test', new_filename)
                img_data = self.prepare_image(img_file_path, (config['img_size'], config['img_size']))
                test_data.append(img_data)
                test_labels.append(norm_attrs)

        # create a list of numpy arrays by attribute
        test_labels_transposed = list(map(list, zip(*test_labels)))
        test_labels_byattr = [np.array(test_labels_transposed[0]), np.array(test_labels_transposed[1]), np.array(test_labels_transposed[2]),
                np.array(test_labels_transposed[3]),
                np.array(test_labels_transposed[4]), np.array(test_labels_transposed[5]), np.array(test_labels_transposed[6])]

        self.X_train = np.array(train_data)
        self.y_train = train_labels
        self.X_test = np.array(test_data)
        self.y_test = test_labels_byattr

    def prepare_image(self, image_path, target_size):
        """Loads image from filepath and scales RGB values from -1 to 1

        Args:
            image_path (str): Image Path
            target_size (tuple): scale to x, y dimensions

        Returns:
            image_array
        """
        img = image.load_img(image_path, target_size = target_size)
        x = image.img_to_array(img)
        x = preprocess_input(x, mode='tf')
        return x


    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test








    