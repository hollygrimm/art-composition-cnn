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
        attributes = [k for k in config['numerical_loss_weights']]
        usecols.extend(attributes)
        categorical_attributes = [k for k in config['categorical_loss_weights']]
        usecols.extend(categorical_attributes)
        df = pd.read_csv(config['train_test_csv_file'], usecols=usecols)

        # train on only those rows that have attribute information
        filtered = df.dropna(subset=[attributes[0]])  

        self.colors_hsv = config['colors']

        # TODO: Train on harmony and style
        # self.harmony_encoder = LabelEncoder()
        # self.harmony_encoder.fit(config['harmonies'])    

        # self.style_encoder = LabelEncoder()
        # self.style_encoder.fit(config['styles'])          

        filenames = []
        labels = []
        for row in filtered.itertuples(index=True, name='Pandas'):
            in_train = getattr(row, "in_train")
            new_filename = getattr(row, "new_filename")

            # normalize attribute values input: [1, 10]   output: [-1, 1]
            min_val = 1
            max_val = 10
            norm_attrs = [(2 * ((getattr(row, attr_name) - min_val)/(max_val - min_val))) - 1 for attr_name in attributes]

            color = getattr(row, 'pri_color')
            hues = 360
            norm_attrs.extend([[np.sin(2*np.pi*self.colors_hsv[color][0]/hues), np.cos(2*np.pi*self.colors_hsv[color][0]/hues), self.colors_hsv[color][1], self.colors_hsv[color][2]]])

            # TODO: Train on harmony and style
            # harmony = self.harmony_encoder.transform([getattr(row, 'harmony')])
            # norm_attrs.extend(to_categorical(harmony, num_classes=len(self.harmony_encoder.classes_)))

            # style = self.style_encoder.transform([getattr(row, 'style')])
            # norm_attrs.extend(to_categorical(style, num_classes=len(self.style_encoder.classes_)))            

            if in_train:
                img_file_path = os.path.join('data/train', new_filename)
            else:
                img_file_path = os.path.join('data/test', new_filename)
            filenames.append(img_file_path)
            labels.append(norm_attrs)

        val_index = int(len(filenames) * (1 - config['validation_split']))

        train_filenames = filenames[:val_index]
        train_labels = labels[:val_index]  
        val_filenames = filenames[val_index:]
        val_labels = labels[val_index:]  

        print("{} training examples".format(len(train_filenames)))
        print("{} validation examples".format(len(val_filenames)))

        self.X_train_filenames = np.array(train_filenames)
        self.y_train = train_labels
        self.X_val_filenames = np.array(val_filenames)
        self.y_val = val_labels     
  

    def get_train_data(self):
        return self.X_train_filenames, self.y_train

    def get_val_data(self):
        return self.X_val_filenames, self.y_val








    