import sys
import unittest
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


sys.path.append('../utils')
from utils import process_config, calc_degrees

class TestArtCompositionCNN(unittest.TestCase):

    def test_encoders(self):
        config, _, _ = process_config('../input_params.json')
        color_encoder = LabelEncoder()
        color_encoder.fit(list(config['colors'].keys()))

        harmony_encoder = LabelEncoder()
        harmony_encoder.fit(config['harmonies'])   

        color = 'blue'
        color_enc = color_encoder.transform([color])
        self.assertListEqual(color_enc.tolist(), [1])
        color_onehot = to_categorical(color_enc, num_classes=len(color_encoder.classes_))
        self.assertListEqual(color_onehot.tolist(), [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        color = 'green-cyan'
        color_enc = color_encoder.transform([color])
        self.assertListEqual(color_enc.tolist(), [6])
        color_onehot = to_categorical(color_enc, num_classes=len(color_encoder.classes_))
        self.assertListEqual(color_onehot.tolist(), [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])        

        color = 'yellow'
        color_enc = color_encoder.transform([color])
        self.assertListEqual(color_enc.tolist(), [12])
        color_onehot = to_categorical(color_enc, num_classes=len(color_encoder.classes_))
        self.assertListEqual(color_onehot.tolist(), [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])        

        harmony = 'analogous'
        harmony_enc = harmony_encoder.transform([harmony])
        self.assertListEqual(harmony_enc.tolist(), [0])
        harmony_onehot = to_categorical(harmony_enc, num_classes=len(harmony_encoder.classes_))
        self.assertListEqual(harmony_onehot.tolist(), [[1, 0, 0, 0, 0, 0]])

        harmony = 'triadic'
        harmony_enc = harmony_encoder.transform([harmony])
        self.assertListEqual(harmony_enc.tolist(), [5])
        harmony_onehot = to_categorical(harmony_enc, num_classes=len(harmony_encoder.classes_))
        self.assertListEqual(harmony_onehot.tolist(), [[0, 0, 0, 0, 0, 1]])        
        
    def test_color_encoder(self):
        config, _, _ = process_config('../input_params.json')
        colors_hsv = config['colors']
        color = 'blue'
        self.assertAlmostEqual(np.sin(2*np.pi*colors_hsv[color][0]/360), -0.8660254037844384)
        self.assertAlmostEqual(np.cos(2*np.pi*colors_hsv[color][0]/360), -0.5000000000000004)
        self.assertListEqual(colors_hsv[color], [240, 1, 1])

    def test_calc_degrees(self):
        degrees = calc_degrees(-.866, -.5)
        self.assertEqual(round(degrees), 240)


if __name__ == '__main__':
    unittest.main()



