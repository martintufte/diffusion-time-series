"""
`Dataset` (pytorch) class is defined.
"""
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

from pathlib import Path


class StandardScaler:
    """
    Simple Encoder/Decoder scaler for standarization of time series.
    Functionally similar to sklearn.preprosessing.StandardScaler, but uses the exact same
    procedure as in UCR data set to standardize, following
    https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries/blob/dcc674541a94ca8a54fbb5503bb75a297a5231cb/ucr.py#L30
    """
    
    def __init__(self, copy=True, with_mean=True, with_std=True):
        self.mean = 0.0
        self.std  = 1.0
        self.with_mean = with_mean
        self.with_std = with_std
    
    def fit(self, X):
        if self.with_mean:    
            self.mean = np.nanmean(X)
        if self.with_std:    
            self.std = math.sqrt(np.nanvar(X))
            self.std = 1.0 if (self.std==0.0) else self.std
    
    def transform(self, X):
        return (X - self.mean) / self.std
        
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        return (X * self.std) + self.mean




class DatasetImporterUCR(object):
    """
    This uses train and test sets as given.
    To compare with the results from ["Unsupervised scalable representation learning for multivariate time series"]
    """
    def __init__(self, cwdir, name:str, standardize_data=True, encode_labels=True, to_tensor_float32=True):
        self.data_root = cwdir.joinpath('datasets', 'UCRArchive_2018', name)
        self.name = name
        
        # fetch an entire dataset
        df_train = pd.read_csv(self.data_root.joinpath(f"{name}_TRAIN.tsv"), sep='\t', header=None)
        df_test = pd.read_csv(self.data_root.joinpath(f"{name}_TEST.tsv"), sep='\t', header=None)
        self.X_train, self.X_test = df_train.iloc[:, 1:].values, df_test.iloc[:, 1:].values
        self.Y_train, self.Y_test = df_train.iloc[:, 0].values,  df_test.iloc[:, 0].values
        
        # standardize data        
        self.standard_scaler = StandardScaler()
        if standardize_data:
            self.X_train = self.standard_scaler.fit_transform(self.X_train)
            self.X_test = self.standard_scaler.transform(self.X_test)
        else:
            self.scaler.fit(self.X_train)
        
        # encode labels
        self.label_encoder = LabelEncoder()
        if encode_labels:
            self.Y_train = self.label_encoder.fit_transform(self.Y_train)[:, None]
            self.Y_test = self.label_encoder.transform(self.Y_test)[:, None]
        else:
            self.label_encoder.fit(self.Y_train)
        
        # reshape from (batch, ts_length) to (batch, channels=1, ts_length)
        self.X_train = np.expand_dims(self.X_train, axis=1)
        self.X_test  = np.expand_dims(self.X_test, axis=1)
        self.Y_train = self.Y_train.reshape(-1, 1)
        self.Y_test = self.Y_test.reshape(-1, 1)
        
        np.nan_to_num(self.X_train, copy=False)
        np.nan_to_num(self.X_test, copy=False)
        
        if to_tensor_float32:
            self.X_train = torch.Tensor(self.X_train).to(torch.float32)
            self.Y_train = torch.Tensor(self.Y_train).to(torch.int32)
            self.X_test = torch.Tensor(self.X_test).to(torch.float32)
            self.Y_test = torch.Tensor(self.Y_test).to(torch.int32)
        
        self.n_classes = self.label_encoder.classes_.size
        self.ts_length = self.X_train.shape[-1]
        
        print('\n Importer:')
        print('X_train:', self.X_train.shape)
        print('X_test:', self.X_test.shape)
        print('Y_train:', self.Y_train.shape)
        print('Y_test:', self.Y_test.shape)

        print("Labels:", self.label_encoder.inverse_transform(np.unique(self.Y_train)))
        print("Encoded labels:", np.unique(self.Y_train))




class UCRDataset(Dataset):
    def __init__(self,
                 kind: str,
                 dataset_importer: DatasetImporterUCR,
                 **kwargs):
        """
        :param kind: "train" / "test"
        :param dataset_importer: instance of the `DatasetImporter` class.
        """
        super().__init__()
        self.kind = kind

        if kind == "train":
            self.X, self.Y = dataset_importer.X_train, dataset_importer.Y_train
        elif kind == "test":
            self.X, self.Y = dataset_importer.X_test, dataset_importer.Y_test
        elif kind == "train/test" or kind == "test/train":
            self.X = torch.concat((dataset_importer.X_train, dataset_importer.X_test), dim=0)
            self.Y = torch.concat((dataset_importer.Y_train, dataset_importer.Y_test), dim=0)
        else:
            raise ValueError

        self._len = self.X.shape[0]

    def getitem_default(self, idx):
        x, y = self.X[idx, :], self.Y[idx, :]
        return x, y

    def __getitem__(self, idx):
        return self.getitem_default(idx)

    def __len__(self):
        return self._len


class SynthethicUCRDataset(Dataset):
    def __init__(self,
                 diffusion_model,
                 reference_ds,
                 sampling_steps = 100,
                 guidance_strength = 0.0,
                 **kwargs):
        """
        :param diffusion_model: instance of the `VDM` class.
        :param reference_ds: instance of the `UCRDataset` class.
        :param sampling_steps: the number of sampling steps
        :param guidance_strength: guidance strength during sampling
        """
        super().__init__()
        self.kind = "synthetic"
        self.sampling_steps = sampling_steps
        self.guidance_strength = guidance_strength
        
        self._len = len(reference_ds)
        
        perm = torch.randperm(self._len)
        self.Y = reference_ds.Y[perm, :]
        self.X = diffusion_model.sample(self._len, sampling_steps, self.Y.view(-1), guidance_strength)
        

    def getitem_default(self, idx):
        x, y = self.X[idx, :], self.Y[idx, :]
        return x, y

    def __getitem__(self, idx):
        return self.getitem_default(idx)

    def __len__(self):
        return self._len


def filter_UCRDatasets(path):
    path = path.joinpath('datasets', 'DataSummary_UCR.csv')
    
    df = pd.read_csv(path)
    df['Filtered'] = True
    
    # remove varying length time series
    df.loc[df['Length'] == 'Vary', 'Filtered'] = False  
    
    # remove time series with less than 100 training examples
    df.loc[df['Train']<100, 'Filtered'] = False
    
    # remove time series with more than 10 classes
    df.loc[df['Class']>10, 'Filtered'] = False
    
    return df # df.loc[df['Filtered'],:].reset_index(drop=True)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    path = Path('C:/Users/marti/OneDrive/Dokumenter/9. semester/Prosjektoppgave/diffusion-time-series')

    # data pipeline
    dataset_importer = DatasetImporterUCR(path, "Yoga", standardize_data=True, encode_labels=True)
    dataset = UCRDataset("train", dataset_importer)
    data_loader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=True)

    # get a mini-batch of samples
    for batch in data_loader:
        x, y = batch
        break
    print('x.shape:', x.shape)

    # plot
    n_samples = 5
    c = 0
    fig, axes = plt.subplots(n_samples, 1, figsize=(3.5, 1.7*n_samples))
    for i, ax in enumerate(axes):
        ax.plot(x[i, c])
    plt.tight_layout()
    plt.show()

    # df
    df = filter_UCRDatasets(path)
    
    
    
    
    
    