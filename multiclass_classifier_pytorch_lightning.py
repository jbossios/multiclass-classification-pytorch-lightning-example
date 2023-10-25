import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform, choice, normal
from torch import nn, optim, Tensor, manual_seed, argmax
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.classification import Accuracy, MulticlassConfusionMatrix
from pytorch_lightning.utilities.model_summary import ModelSummary
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Set seeds for reproducibility
import random
random.seed(42)
manual_seed(42)
np.random.seed(42)


def create_data(
        k: int,  # number of classes
        npoints: int,  # total number of points to be generated
        seed: int = 1  # seed for reproducibility
    ) -> tuple[np.array, np.array]:
    """
    Randomly create npoints data points classified into k classes
    """
    # Set seed for reproducibility
    np.random.seed(seed)
    X = []  # collect data
    labels = []  # collect class labels
    
    # Randomly define a centroid for each class
    x_centers = {i: uniform(30, 70) for i in range(k)}
    y_centers = {i: uniform(0, 400000) for i in range(k)}
    
    # Generate npoints
    for ipoint in range(npoints):
        # Randomly assign this point to a class
        ik = choice(range(k))
        labels.append(ik)
        
        # Retrieve centroid for this class
        center_x = x_centers[ik]
        center_y = y_centers[ik]
        
        # Generate point
        X.append([
            normal(center_x, 3),
            normal(center_y, 15000)
        ])

    return np.array(X), np.array(labels)


def make_scatter_plot(
        data: np.array,  # 2D array
        labels: np.array,  # 1D array with class labels
        k: int,  # number of classes
        outname: str  # output name
    ) -> None:
    """ Make scatter plot"""
    
    # Protection
    assert k < 9, "Only up to 8 classes are supported"
    
    # Define colors for each class
    colors = {
        0: 'red',
        1: 'black',
        2: 'blue',
        3: 'green',
        4: 'yellow',
        5: 'orange',
        6: 'magenta',
        7: 'cyan',
    }

    # Plot data from each class
    title = outname.split('.')[0]
    plt.figure(title)
    plt.title(title)
    for ik in range(k):  # loop over classes
        # Get data for class ik
        data_k = data[labels == ik]

        # Unpack data
        X = list(zip(*data_k))
        x, y = X[0], X[1]

        # Make scatter plot and add x- and y-axis titles
        plt.scatter(x, y, c = colors[ik])

    # Save figure
    plt.savefig(outname)


def standardize_data(data: tuple[np.array, np.array]) -> tuple[np.array, np.array]:
    """ Standardize data using StandardScaler """
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)
    return data_standardized


class Model(pl.LightningModule):
    def __init__(self, k, optimizer = 'Adam', dropout_rate = 0):
        super().__init__()
        # Set list of layers in the model
        layers = [nn.Linear(2, 16), nn.ReLU()]
        if dropout_rate != 0:
            layers.append(nn.Dropout(dropout_rate))
        layers.extend([nn.Linear(16, 8), nn.ReLU()])
        if dropout_rate != 0:
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(8, k))
        # Create model
        self.model = nn.Sequential(*layers)
        # Define other attributes
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.lr = {'Adam': 0.001, 'SGD': 0.1}[optimizer]
        self.accuracy = Accuracy(task="multiclass", num_classes=k)
        self.test_pred = []  # collect predictions
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=k)
        
    def forward(self, x):
        return self.model(x)
   
    def configure_optimizers(self):
        if self.optimizer == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        else:
            optimizer = optim.SGD(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log('loss', loss)
        # Track accuracy
        y_target = argmax(y, dim=1)
        y_pred = argmax(logits, dim=1)
        acc = self.accuracy(y_pred, y_target)
        self.log('accuracy', acc)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log('val_loss', loss)
        # Track accuracy
        y_target = argmax(y, dim=1)
        y_pred = argmax(logits, dim=1)
        acc = self.accuracy(y_pred, y_target)
        self.log('val_accuracy', acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        # Evaluate model
        logits = self.forward(x)
        # Track loss
        loss = self.loss(logits, y)
        self.log('test_loss', loss)
        # Track accuracy
        y_target = argmax(y, dim=1)
        y_pred = argmax(logits, dim=1)  # find label with highest probability
        acc = self.accuracy(y_pred, y_target)
        self.log('test_accuracy', acc)
        # Collect predictions
        self.test_pred.extend(y_pred.cpu().numpy())
        # Update confusion matrix
        self.confusion_matrix.update(y_pred, y_target)
        

class DataModule(pl.LightningDataModule):
    def __init__(self, data_dict: dict, batch_size: int = 32):
        super().__init__()
        self.data_dict = data_dict
        self.batch_size = batch_size

    def train_dataloader(self):
        X = Tensor(self.data_dict['training'][0])
        y = Tensor(self.data_dict['training'][1])
        tensor_dataset = TensorDataset(X, y)
        return DataLoader(tensor_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        X = Tensor(self.data_dict['testing'][0])
        y = Tensor(self.data_dict['testing'][1])
        tensor_dataset = TensorDataset(X, y)
        return DataLoader(tensor_dataset, batch_size=self.batch_size)
    
    def val_dataloader(self):
        X = Tensor(self.data_dict['validation'][0])
        y = Tensor(self.data_dict['validation'][1])
        tensor_dataset = TensorDataset(X, y)
        return DataLoader(tensor_dataset, batch_size=self.batch_size)


class MetricTrackerCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.losses = {
            'loss': [],
            'val_loss': []
        }
        self.acc = {
            'accuracy': [],
            'val_accuracy': []
        }

    def on_train_epoch_end(self, trainer, module):
        metrics = trainer.logged_metrics
        self.losses['loss'].append(metrics['loss'])
        self.acc['accuracy'].append(metrics['accuracy'])

    def on_validation_epoch_end(self, trainer, module):
        metrics = trainer.logged_metrics
        self.losses['val_loss'].append(metrics['val_loss'])
        self.acc['val_accuracy'].append(metrics['val_accuracy'])


def plot_loss(loss_dict) -> None:
    """ Plot loss and val_loss """
    plt.figure('loss')
    plt.plot(loss_dict['loss'], label='loss', c='black')
    plt.plot(loss_dict['val_loss'], label='val_loss', c='red')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('loss.png')


def plot_accuracy(acc_dict) -> None:
    """ Plot accuracy and val_accuracy """
    plt.figure('accuracy')
    plt.plot(acc_dict['accuracy'], label='accuracy', c='black')
    plt.plot(acc_dict['val_accuracy'], label='val_accuracy', c='red')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('accuracy.png')


def compare_distributions(hists, k) -> None:
    """ Compare distributions """
    
    # Protection
    assert k < 9, "Only up to 8 classes are supported"

    # Define a color for every class
    colors = {
        0: 'red',
        1: 'black',
        2: 'blue',
        3: 'green',
        4: 'yellow',
        5: 'orange',
        6: 'magenta',
        7: 'cyan',
    }
    
    # List of data categories
    data_categories = hists.keys()
    
    # Prepare the data
    data_dict = {}
    for i, (data_type, data) in enumerate(hists.items()):
        _, values = np.unique(data, return_counts=True)  # get counts for every class
        data_dict[data_type] = values
 
    # Create numerical axis
    x = np.arange(k)

    # Set the width of the bars
    width = 0.2

    # Create figure and axis
    fig, ax = plt.subplots()

    # Create bars
    for i, (data_type, counts) in enumerate(data_dict.items()):
        offset = width * i  # offset in the x-axis (different for each bar)
        bar = ax.bar(x + offset, counts, width=width, label=data_type, color=colors[i])
        ax.bar_label(bar)  # show numbers on top of bars

    # Label the axes
    ax.set_xlabel('Class')
    ax.set_ylabel('Counts')

    # Show category names
    ax.set_xticks(x + width, list(range(k)))

    # Add legends
    ax.legend()

    # Save figure
    fig.savefig('compare_distribution_of_classes_data.png')


def to_numerical(labels):
    """ Revert one-hot encoding """
    return np.argmax(labels, axis=1)


def main(
        k: int,  # number of classes
        train_model: bool  # train model or load previous trained model
    ) -> None:
    
    # Protections
    if not train_model and not os.path.exists('best_model.ckpt'):
        print('ERROR: best_model.ckpt file can not be found, run without the --loadBestModel flag for training a model first, exiting')
        sys.exit(1)
    elif train_model and os.path.exists('best_model.ckpt'):
        print('INFO: best_model.ckpt will be deleted...')
        os.remove('best_model.ckpt')
    
    # Create fake data
    data, labels = create_data(k=k, npoints=100)
    make_scatter_plot(data=data, labels=labels, k=k, outname='data.png')

    # Convert labels using one-hot encoding technique
    labels = labels.reshape(-1, 1)
    labels = OneHotEncoder(sparse_output=False).fit_transform(labels)
    
    # Separate data into training, validation and testing data
    #   30% is used for testing
    #   20% of the remaining data is used for validation
    #   the rest is used for training
    x_train_tmp, X_test, y_train_tmp, y_test = train_test_split(data, labels, test_size=0.3, random_state=10)
    X_train, X_val, y_train, y_val = train_test_split(x_train_tmp, y_train_tmp, test_size=0.2, random_state=10)

    # Compare distribution of classes
    compare_distributions(
            {
                'all': to_numerical(labels),
                'training': to_numerical(y_train),
                'testing': to_numerical(y_test),
                'validation': to_numerical(y_val)
            },
            k
        ) 

    # Standardize features in all datasets
    data_dict = {
        'training': (X_train, y_train),
        'testing': (X_test, y_test),
        'validation': (X_val, y_val),
    }
    standardized_data_dict = {}
    for dataset_type, data_tuple in data_dict.items():
        data_standardized = standardize_data(data=data_tuple[0])
        standardized_data_dict[dataset_type] = (data_standardized, data_tuple[1])

    # Visualize standardized features in training dataset
    make_scatter_plot(
        data = standardized_data_dict['training'][0],
        labels = to_numerical(standardized_data_dict['training'][1]),
        k = k,
        outname = 'standardized_training_data.png'
    )

    # Create an instance of our model
    model = Model(k=k)
    summary = ModelSummary(model, max_depth=-1)
    print(summary)

    # Create a PyTorch Lightning trainer and add callbacks
    tracker = MetricTrackerCallback()
    early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor = 'val_loss',
        patience = 10,
        min_delta = 0.005,
        mode = 'min',
    )
    dirpath = os.path.dirname(__file__)  # current path
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath = dirpath,
        filename = 'best_model',
        monitor = 'val_loss',
        mode = 'min',
    )
    trainer = pl.Trainer(
        max_epochs = 300,
        enable_model_summary = False,  # summary printed already
        callbacks = [
            tracker,
            early_stopping_callback,
            model_checkpoint_callback
        ]
    )

    # Create an instance of the data module
    data_module = DataModule(data_dict=standardized_data_dict)

    # Train the model (if requested)
    if train_model:
        trainer.fit(model, data_module)
        # Plot metrics
        plot_loss(tracker.losses)  # plot loss vs epoch
        plot_accuracy(tracker.acc)  # plot accuracy vs epoch
    
    # Evaluate model in test data and print accuracy
    result = trainer.test(model, data_module, ckpt_path="best_model.ckpt")
    print(f"Accuracy in test: {result[0]['test_accuracy']}")

    # Get predicted labels on test data
    labels_predicted = model.test_pred

    # Visualize test dataset with true labels
    make_scatter_plot(
        data = data_dict['testing'][0],  # data for testing
        labels = to_numerical(data_dict['testing'][1]),  # labels for testing
        k = k,
        outname = 'test_data.png'
    )

    # Visualize test dataset with predicted labels
    make_scatter_plot(
        data = data_dict['testing'][0],  # data for testing
        labels = np.array(labels_predicted),  # predicted labels for test data
        k = k,
        outname = 'test_data_predicted_labels.png'
    )

    # Plot confusion matrix
    fig, _ = model.confusion_matrix.plot()
    fig.savefig('confusion_matrix.png')

    print('>>> ALL DONE <<<')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', '-k', type=int, action='store', default=3, help='Number of classes')
    parser.add_argument('--loadBestModel', action='store_true', default=False, help='No model will be trained and will load model from best_model.ckpt')
    args = parser.parse_args()
    main(args.k, not args.loadBestModel)