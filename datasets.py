import numpy as np
import tensorflow as tf 

def get_fmnist_binary():
    # Consider using Pytorch version of this. 
    # If moving entirely to Pytorch, doesn't make sense to only keep this to get data from fashion_mnist

    (train_x, train_y), _ = tf.keras.datasets.fashion_mnist.load_data() 
    # Pre-process data into binary dataset 
    train_inds = np.where(train_y < 2)[0]  

    train_x = train_x[train_inds] / 255. - 0.5

    train_y = np.eye(2)[train_y[train_inds]]

    # subsample dataset
    ss_inds = np.random.choice(train_x.shape[0], train_x.shape[0]//2, replace=False)
    train_x = train_x[ss_inds]
    train_y = train_y[ss_inds]

    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1] * train_x.shape[2]))

    return (train_x, train_y)

def get_data(dataset="fashion_mnist"):
    datasets = {
        "fashion_mnist" : get_fmnist_binary
    }

    return datasets[dataset]()