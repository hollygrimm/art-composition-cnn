import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from base.base_data_loader import BaseDataLoader
from keras.utils import to_categorical

class WikiArtDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(WikiArtDataLoader, self).__init__(config)

        # read only the necessary columns from the CSV
        usecols = ['in_train', 'new_filename']
        attributes = [k for k in config['loss_weights'] if k not in config['categorical_attrs']]
        usecols.extend(attributes)
        categorical_attributes = [k for k in config['categorical_attrs']]
        usecols.extend(categorical_attributes)
        df = pd.read_csv(config['train_test_csv_file'], usecols=usecols)

        train_filenames = []
        train_labels = []
        test_filenames = []
        test_labels = []    

        # train on only those rows that have attribute information
        filtered = df.dropna(subset=[attributes[0]])  

        # create encoder from class values to integers
        self.color_encoder = LabelEncoder()
        # FIXME pass custom list of colors
        self.color_encoder.fit(filtered['pri_color'])
        print(self.color_encoder.classes_)

        self.harmony_encoder = LabelEncoder()
        # FIXME pass custom list of harmonies
        self.harmony_encoder.fit(filtered['harmony'])            

        for row in filtered.itertuples(index=True, name='Pandas'):
            in_train = getattr(row, "in_train")
            new_filename = getattr(row, "new_filename")

            # normalize attribute values input: [1, 10]   output: [-1, 1]
            min_val = 1
            max_val = 10
            norm_attrs = [(2 * ((getattr(row, attr_name) - min_val)/(max_val - min_val))) - 1 for attr_name in attributes]

            color = self.color_encoder.transform([getattr(row, 'pri_color')])
            norm_attrs.extend(to_categorical(color, num_classes=len(self.color_encoder.classes_)))
            harmony = self.harmony_encoder.transform([getattr(row, 'harmony')])
            norm_attrs.extend(to_categorical(harmony, num_classes=len(self.harmony_encoder.classes_)))

            if in_train:
                img_file_path = os.path.join('data/train', new_filename)
                train_filenames.append(img_file_path)
                train_labels.append(norm_attrs)
            else:
                img_file_path = os.path.join('data/test', new_filename)
                test_filenames.append(img_file_path)
                test_labels.append(norm_attrs)
        print("{} training examples".format(len(train_filenames)))
        print("{} test examples".format(len(test_filenames)))

        self.X_train_filenames = np.array(train_filenames)
        self.y_train = train_labels
        self.X_test_filenames = np.array(test_filenames)
        self.y_test = test_labels


    def get_train_data(self):
        return self.X_train_filenames, self.y_train

    def get_test_data(self):
        return self.X_test_filenames, self.y_test








    