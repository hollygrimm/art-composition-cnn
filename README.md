# ResNet50 on Art Composition Attributes

Please read the accompanying blog post: [https://hollygrimm.com/acan_final](https://hollygrimm.com/acan_final)

## Requirements
Keras version 2.1.2

Keras 2.1.6 errors out with:
```
'int' object has no attribute 'ndim'
```
https://github.com/jacobgil/keras-dcgan/issues/23

## Run Training
```
python main.py -c input_params.json
```

## Run Inference on Validation Samples
```
python
import main
main.infer()
```

## Run Tests
```
cd tests
python art_composition_cnn_tests.py
```