from DataProcesser import DataProcesser 
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

a = np.zeros((10, 1), dtype=int)
b = np.random.randint(size=(10, 9), low=0, high=6)
c = np.zeros((10, 10), dtype=int)

data = np.concatenate([a, b, c], axis=1)
processer = DataProcesser(df=data)
user_to_index, movie_to_index, df_to_index = processer.transform_to_index()
loader = CustomLoader(data=df_to_index, test_size=0.1, batch_size=16)
train_loader = loader.get_train_loader()
test_loader = loader.get_test_loader()
model = MatrixFactorization(num_users=len(user_to_index), num_movies=len(movie_to_index))
trainer = Trainer(model=model, train_loader=train_loader, test_loader=test_loader, epochs=10)
trainer.train()
trainer.test()
predictor = Predictor(model=model, user_to_index=user_to_index, movie_to_index=movie_to_index, df_to_index=df_to_index)
print(predictor.predict())
service = Service(data=data)
service.train()
service.test()
print(service.predict())