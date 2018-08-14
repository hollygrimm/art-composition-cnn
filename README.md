# ResNet50 on Art Composition Attributes

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
