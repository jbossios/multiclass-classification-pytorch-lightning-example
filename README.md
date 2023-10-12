# Multiclass classifier with deep neural networks using PyTorch Lightning

## Dependencies
```
numpy
matplotlib
torch
torchmetrics
pytorch-lightning
sklearn
```

## Introduction

In ```multiclass_classifier_pytorch_lightning.py```, you can find an example on how to implement and train a multiclass classifier based on deep neural networks with PyTorch Lightning, and how to evaluate its performance.

This example uses fake data, generated randomly. This data is characterized by two features and is classified into ```k``` labels. In this example, we will learn those labels.

## Run

For generating the data, training the model and evaluating the performance in testing data, run the following:


```
python multiclass_classifier_pytorch_lightning.py
```

Note:
- You can choose the number of classes with the ```-k``` flag (if not provided, 3 classes will be used)

The above script will save the best trained model to ```best_model.ckpt``` and will create eight PNG images:

- ```data.png```: this is a scatter plot with the generated data, colored by the corresponding label
- ```standardized_training_data.png```:  this shows the data that will be used for training, which was already standardized
- ```test_data.png```:  this one shows the data that is used for testing
- ```test_data_predicted_labels.png```: the same test data is showed but data is colored based on the predicted labels (if the DNN works well it should look very similar to ```test_data.png```)
- ```loss.png```: this will have the (training) loss and validation loss (val_loss) vs epoch
- ```accuracy.png```: this will have the (training) accuracy and validation accuracy (val_accuracy) vs epoch
- ```compare_distribution_of_classes_data.png```: this compares the distribution of classes for each dataset type (training, test, validation, all=complete dataset)
- ```confusion_matrix.png```: as the name suggests, this will have the confusion matrix using the test data

If you like this example, please consider giving me a star!
