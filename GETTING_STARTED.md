# Getting Started with CA3

This document provides a brief intro of launching jobs in PySlowFast for training and testing. Before launching any job, make sure you have properly installed the settings following the instruction in [README.md](README.md) and you have prepared the dataset following [DATASET.md](DATASET.md) with the correct format.

## Train a Standard Model from Scratch

Here we can start with training the models by running:

```
python main.py
```
You may need to pass location of your dataset in the command line by adding `DATA.PATH_TO_DATA_DIR path_to_your_dataset`.

## Perform Test
We have `TRAIN.ENABLE` and `TEST.ENABLE` to control whether training or testing is required for the current job. If only testing is preferred, you can run the test.py.
```
python test.py \
```


