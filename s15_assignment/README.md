# Transformer-Based Machine Translation

This repository contains the code for training a transformer-based machine translation model using PyTorch and PyTorch Lightning. The project is structured into several files:

[model.py](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s15_assignment/model.py): Defines the transformer architecture for machine translation.

[config.py](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s15_assignment/config.py): Contains configuration parameters for the training process.

[dataset.py](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s15_assignment/dataset.py): Implements the dataset loader and preprocessing for machine translation data.

[train.py](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s15_assignment/train.py): Contains the training loop and logic for training the machine translation model.

[custom_transformer_lightning.py](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s15_assignment/custom_transformer_lightning.py): Implements a custom PyTorch Lightning module for training and evaluation.

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

## Training Log
Final Train Loss: 3.7000
![Loss](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s15_assignment/images/val_loss.jpg)

```
  | Name    | Type             | Params
---------------------------------------------
0 | model   | Transformer      | 75.1 M
1 | loss_fn | CrossEntropyLoss | 0     
---------------------------------------------
75.1 M    Trainable params
0         Non-trainable params
75.1 M    Total params
300.532   Total estimated model params size (MB)
Max length of source sentence: 309
Max length of target sentence: 274
    SOURCE: I will only go and change . Order tea .'
    TARGET: Fa’ portare il tè.
 PREDICTED: echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare benedicevo benedicevo benedicevo benedicevo echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare concederà concederà benedicevo benedicevo benedicevo benedicevo concederà concederà concederà benedicevo sghembo echeggiare echeggiare concederà concederà concederà echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare concederà concederà concederà concederà echeggiare echeggiare echeggiare echeggiare echeggiare benedicevo benedicevo benedicevo benedicevo benedicevo echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo echeggiare echeggiare echeggiare echeggiare echeggiare benedicevo benedicevo benedicevo benedicevo benedicevo echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare benedicevo benedicevo benedicevo benedicevo echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo benedicevo


    SOURCE: Throughout the trip , he had manifested great curiosity concerning the kettle .
    TARGET: In tutta l’escursione, esso aveva mostrato una grande curiosità riguardo al calderino.
 PREDICTED: echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare benedicevo benedicevo benedicevo echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare intendiamoci intendiamoci intendiamoci intendiamoci echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare echeggiare benedicevo benedicevo benedicevo benedicevo echeggiare echeggiare echeggiare echeggiare benedicevo benedicevo benedicevo echeggiare


/usr/local/lib/python3.10/dist-packages/torchmetrics/utilities/prints.py:62: FutureWarning: Importing `CharErrorRate` from `torchmetrics` was deprecated and will be removed in 2.0. Import `CharErrorRate` from `torchmetrics.text` instead.
  _future_warning(
/usr/local/lib/python3.10/dist-packages/torchmetrics/utilities/prints.py:62: FutureWarning: Importing `WordErrorRate` from `torchmetrics` was deprecated and will be removed in 2.0. Import `WordErrorRate` from `torchmetrics.text` instead.
  _future_warning(
/usr/local/lib/python3.10/dist-packages/torchmetrics/utilities/prints.py:62: FutureWarning: Importing `BLEUScore` from `torchmetrics` was deprecated and will be removed in 2.0. Import `BLEUScore` from `torchmetrics.text` instead.
  _future_warning(
Max length of source sentence: 309
Max length of target sentence: 274
Epoch 9: 100%
3638/3638 [32:33<00:00, 1.86it/s, v_num=0, train_loss=3.700]
    SOURCE: I will only go and change . Order tea .'
    TARGET: Fa’ portare il tè.
 PREDICTED: Non è un momento di , ma non è un ’ altra .


    SOURCE: Throughout the trip , he had manifested great curiosity concerning the kettle .
    TARGET: In tutta l’escursione, esso aveva mostrato una grande curiosità riguardo al calderino.
 PREDICTED: Il suo momento era un momento , e .


    SOURCE: I will only go and change . Order tea .'
    TARGET: Fa’ portare il tè.
 PREDICTED: Io mi e .


    SOURCE: Throughout the trip , he had manifested great curiosity concerning the kettle .
    TARGET: In tutta l’escursione, esso aveva mostrato una grande curiosità riguardo al calderino.
 PREDICTED: E il suo tempo era un ’ altra cosa , che era un ’ altra volta .


    SOURCE: I will only go and change . Order tea .'
    TARGET: Fa’ portare il tè.
 PREDICTED: Io sono molto contenta di andare a sedere .


    SOURCE: Throughout the trip , he had manifested great curiosity concerning the kettle .
    TARGET: In tutta l’escursione, esso aveva mostrato una grande curiosità riguardo al calderino.
 PREDICTED: il tempo , era stato stato in un momento .


    SOURCE: I will only go and change . Order tea .'
    TARGET: Fa’ portare il tè.
 PREDICTED: Io sono molto contento di andare a prendere il tè .


    SOURCE: Throughout the trip , he had manifested great curiosity concerning the kettle .
    TARGET: In tutta l’escursione, esso aveva mostrato una grande curiosità riguardo al calderino.
 PREDICTED: In quel momento , dopo , era un po ’ di , si .


    SOURCE: I will only go and change . Order tea .'
    TARGET: Fa’ portare il tè.
 PREDICTED: Io andrò a chiamare il tè .


    SOURCE: Throughout the trip , he had manifested great curiosity concerning the kettle .
    TARGET: In tutta l’escursione, esso aveva mostrato una grande curiosità riguardo al calderino.
 PREDICTED: In quel momento , egli aveva cominciato a la roba , la .


    SOURCE: I will only go and change . Order tea .'
    TARGET: Fa’ portare il tè.
 PREDICTED: Io andrò a prendere il tè .


    SOURCE: Throughout the trip , he had manifested great curiosity concerning the kettle .
    TARGET: In tutta l’escursione, esso aveva mostrato una grande curiosità riguardo al calderino.
 PREDICTED: Per la fine , il viaggio fu , il .


    SOURCE: I will only go and change . Order tea .'
    TARGET: Fa’ portare il tè.
 PREDICTED: Andrò solo a prendere il tè .


    SOURCE: Throughout the trip , he had manifested great curiosity concerning the kettle .
    TARGET: In tutta l’escursione, esso aveva mostrato una grande curiosità riguardo al calderino.
 PREDICTED: Prima , dopo , egli aveva fatto il lavoro di .


    SOURCE: I will only go and change . Order tea .'
    TARGET: Fa’ portare il tè.
 PREDICTED: Andrò da loro e io devo dare un ’ occhiata al tè .


    SOURCE: Throughout the trip , he had manifested great curiosity concerning the kettle .
    TARGET: In tutta l’escursione, esso aveva mostrato una grande curiosità riguardo al calderino.
 PREDICTED: Per quanto fosse stato il più piccolo , la sua impazienza della mostarda .


    SOURCE: I will only go and change . Order tea .'
    TARGET: Fa’ portare il tè.
 PREDICTED: Io andrò a prendere il tè e la serva .


    SOURCE: Throughout the trip , he had manifested great curiosity concerning the kettle .
    TARGET: In tutta l’escursione, esso aveva mostrato una grande curiosità riguardo al calderino.
 PREDICTED: Prima di tutto , egli aveva una buona quantità di carta con la carta .


    SOURCE: I will only go and change . Order tea .'
    TARGET: Fa’ portare il tè.
 PREDICTED: Andrò da solo e parleremo del tè .


    SOURCE: Throughout the trip , he had manifested great curiosity concerning the kettle .
    TARGET: In tutta l’escursione, esso aveva mostrato una grande curiosità riguardo al calderino.
 PREDICTED: Per tutta la stagione asciutta , aveva una buona quantità di storia .


INFO:pytorch_lightning.utilities.rank_zero:`Trainer.fit` stopped: `max_epochs=10` reached.
```
