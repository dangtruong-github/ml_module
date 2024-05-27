from .data_processor import DataProcesser 
from CustomLoader import CustomLoader 
from MatrixFactorization import MatrixFactorization 
from Trainer import Trainer
from Predictor import Predictor
from Service import Service 

import pandas as pd
import numpy as np

import torch
from torch import nn

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader

a = [
  [2, 2, 1], 
  [4, 1, 2], 
  [6, 0, 5]
]
a = pd.DataFrame(a, columns=['user_id', 'movie_id', 'rating'])

service = Service(a, epochs=20)
service.train()
print(service.predict()) 
print(service.predict().shape)