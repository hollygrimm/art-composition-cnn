import sys
import unittest
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


sys.path.append('../utils')
from utils import process_config

class TestArtCompositionCNN(unittest.TestCase):

    def test_encoders(self):
        config, _, _ = process_config('../input_params.json')
        color_encoder = LabelEncoder()
        color_encoder.fit(config['colors'])

        harmony_encoder = LabelEncoder()
        harmony_encoder.fit(config['harmonies'])   

        color = 'orange'
        color_enc = color_encoder.transform([color])
        self.assertListEqual(color_enc.tolist(), [9])
        color_onehot = to_categorical(color_enc, num_classes=len(color_encoder.classes_))
        self.assertListEqual(color_onehot.tolist(), [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])

        harmony = 'analogous'
        harmony_enc = harmony_encoder.transform([harmony])
        self.assertListEqual(harmony_enc.tolist(), [0])
        harmony_onehot = to_categorical(harmony_enc, num_classes=len(harmony_encoder.classes_))
        self.assertListEqual(harmony_onehot.tolist(), [[1, 0, 0, 0, 0, 0]])
        

if __name__ == '__main__':
    unittest.main()



