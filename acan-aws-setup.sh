#!/bin/bash
git clone https://github.com/hollygrimm/art-composition-cnn.git
cd art-composition-cnn/
cd data

# download wikiart 256 test and train examples and untar
wget https://github.com/zo7/painter-by-numbers/releases/download/data-v1.0/train.tgz
wget https://github.com/zo7/painter-by-numbers/releases/download/data-v1.0/test.tgz

if ! echo "82c456e5c66faff41a1806852ed5f42e366c402aa0ae539413079dd65866b3d7  train.tgz" | sha256sum -c -; then
    echo "Checksum failed" >&2
    exit 1
fi

if ! echo "7368fad4950616d74dc04eeb204d6a46dda4c02f50067a5d792ab6365bee1b38  test.tgz" | sha256sum -c -; then
    echo "Checksum failed" >&2
    exit 1
fi

tar -xvf test.tgz
tar -xvf train.tgz
rm -rf test.tgz
rm -rf train.tgz

cd ~
source activate tensorflow_p36
pip install keras
pip install scikit-learn
pip install pillow
source deactivate
