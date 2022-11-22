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





class DatasetImporterUCR(object):
    """
    This uses train and test sets as given.
    To compare with the results from ["Unsupervised scalable representation learning for multivariate time series"]
    """
    def __init__(self, cwdir, subset_name: str, data_scaling, to_tensor_float32=True, **kwargs):
        self.data_root = cwdir.joinpath('datasets', 'UCRArchive_2018', subset_name)

        # fetch an entire dataset
        df_train = pd.read_csv(self.data_root.joinpath(f"{subset_name}_TRAIN.tsv"), sep='\t', header=None)
        df_test = pd.read_csv(self.data_root.joinpath(f"{subset_name}_TEST.tsv"), sep='\t', header=None)

        self.X_train, self.X_test = df_train.iloc[:, 1:].values, df_test.iloc[:, 1:].values
        self.Y_train, self.Y_test = df_train.iloc[:, [0]].values, df_test.iloc[:, [0]].values
        
        le = LabelEncoder()
        self.Y_train = le.fit_transform(self.Y_train)[:, None]
        self.Y_test = le.transform(self.Y_test)[:, None]

        if data_scaling:
            # following [https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries/blob/dcc674541a94ca8a54fbb5503bb75a297a5231cb/ucr.py#L30]
            mean = np.nanmean(self.X_train)
            var = np.nanvar(self.X_train)
            self.X_train = (self.X_train - mean) / math.sqrt(var)
            self.X_test = (self.X_test - mean) / math.sqrt(var)

        # reshape from (batch, ts_length) to (batch, channels=1, ts_length)
        self.X_train = np.expand_dims(self.X_train, axis=1)
        self.X_test  = np.expand_dims(self.X_test, axis=1)
        
        np.nan_to_num(self.X_train, copy=False)
        np.nan_to_num(self.X_test, copy=False)
        
        if to_tensor_float32:
            self.X_train = torch.Tensor(self.X_train).to(torch.float32)
            self.Y_train = torch.Tensor(self.Y_train).to(torch.int32)
            self.X_test = torch.Tensor(self.X_test).to(torch.float32)
            self.Y_test = torch.Tensor(self.Y_test).to(torch.int32)
        

        print('self.X_train.shape:', self.X_train.shape)
        print('self.X_test.shape:', self.X_test.shape)

        print("# unique labels (train):", np.unique(self.Y_train))
        print("# unique labels (test):", np.unique(self.Y_test))




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



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    cwdir = Path('C:/Users/marti/OneDrive/Dokumenter/9. semester/Prosjektoppgave/diffusion-time-series')

    # data pipeline
    dataset_importer = DatasetImporterUCR(cwdir, "Wafer", data_scaling=True)
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




