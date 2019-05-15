# ResNet50 on Art Composition Attributes

Fine-tunes a ResNet50 (pretrained on imagenet) network by training on WikiArt images labeled with eight art composition attributes.
Used with [https://github.com/hollygrimm/cyclegan-keras-art-attrs](https://github.com/hollygrimm/cyclegan-keras-art-attrs) to generate art.

Please read the accompanying blog post: [https://hollygrimm.com/acan_final](https://hollygrimm.com/acan_final)

## Requirements
* keras
* scikit-learn
* pillow

## AWS Install
* Select Deep Learning AMI (Ubuntu) Version 13.0
* Instance Type `GPU Compute` such as p2.xlarge
* 125GB sda1

Connect to instance, copy contents of [acan-aws-setup.sh](acan-aws-setup.sh) to file in /home/ubuntu and run:
```
vi acan-aws-setup.sh
chmod +x acan-aws-setup.sh
./aws-setup.sh
```

## Manual Install

### Download Dataset
download test.tgz and train.tgz from [https://github.com/zo7/painter-by-numbers/releases/tag/data-v1.0](https://github.com/zo7/painter-by-numbers/releases/tag/data-v1.0)

```
cd data
tar -xvf test.tgz
tar -xvf train.tgz
```

## Label Data with Attributes
Example attribute data has been supplied for four examples in [all_domain.csv](data/all_domain.csv). For best results, modify all_domain.csv and label more images with attributes.


## Run Training
```
source activate tensorflow_p36
cd art-composition-cnn/
python main.py -c input_params.json
```

# Tensorboard
```
source activate tensorflow_p36
cd art-composition-cnn/experiments/
tensorboard --logdir=.
```

## Run Inference on Validation Samples
Update weights_path with selected hdf5 from training:
```
vi input_params_for_inference.json
```

Run inference:
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