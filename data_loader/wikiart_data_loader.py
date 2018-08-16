import os
import numpy as np
import pandas as pd
from base.base_data_loader import BaseDataLoader

class WikiArtDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(WikiArtDataLoader, self).__init__(config)

        # read only the necessary columns from the CSV
        usecols = ['in_train', 'new_filename']
        attributes = [k for k in config['loss_weights']]
        usecols.extend(attributes)
        df = pd.read_csv(config['train_test_csv_file'], usecols=usecols)

        train_filenames = []
        train_labels = []
        test_filenames = []
        test_labels = []    

        # train on only those rows that have attribute information
        filtered = df.dropna(subset=[attributes[0]])  

        for row in filtered.itertuples(index=True, name='Pandas'):
            in_train = getattr(row, "in_train")
            new_filename = getattr(row, "new_filename")

            # normalize attribute values input: [1, 10]   output: [-1, 1]
            min_val = 1
            max_val = 10
            norm_attrs = [(2 * ((getattr(row, attr_name) - min_val)/(max_val - min_val))) - 1 for attr_name in attributes]

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








    