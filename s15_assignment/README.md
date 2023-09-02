# Transformer-Based Machine Translation

This repository contains the code for training a transformer-based machine translation model using PyTorch and PyTorch Lightning. The project is structured into several files:

[model.py](): Defines the transformer architecture for machine translation.

[config.py](): Contains configuration parameters for the training process.

[dataset.py](): Implements the dataset loader and preprocessing for machine translation data.

[train.py](): Contains the training loop and logic for training the machine translation model.

[custom_transformer_lightning.py](): Implements a custom PyTorch Lightning module for training and evaluation.

## Model Architecture (model.py)
The model.py file defines the transformer architecture used for machine translation. It includes components such as:

- Positional encoding
- Multi-head attention
- Feedforward layers
- Encoder and decoder blocks
- Layer normalization

## Configuration (config.py)
The config.py file stores configuration parameters for the training process. You can adjust these parameters to control aspects like batch size, learning rate, model size, and training epochs. Additionally, you can specify whether to preload a pre-trained model checkpoint.

## Dataset Loading (dataset.py)
The dataset.py file handles dataset loading and preprocessing. It defines a custom dataset class, BilingualDataset, that reads and preprocesses bilingual text data. This class also tokenizes the input sentences and generates masks for padding.

## Training Loop (train.py)
The train.py file contains the main training loop for training the machine translation model. It uses PyTorch Lightning to streamline the training process. You can run this script to train your model using the specified configuration.

## Custom Lightning Module (custom_transformer_lightning.py)
The custom_transformer_lightning.py file implements a custom PyTorch Lightning module named CustomTransformer. This module inherits from pl.LightningModule and defines the training and validation logic. It also includes a method for greedy decoding during validation to generate translations.
