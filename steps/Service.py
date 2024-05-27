import numpy as np 
import pandas as pd 

from .data_processor import DataProcesser 
from .CustomLoader import CustomLoader 
from .MatrixFactorization import MatrixFactorization 
from .Trainer import Trainer
from .Predictor import Predictor 

class Service:
  def __init__(
      self,
      data,
      epochs=10
  ):

    """
      Args: data is np.array from database
    """
    self.data = data

    self.processer = DataProcesser(self.data)
    self.user_to_index, self.movie_to_index, self.df_to_index=self.processer.transform_to_index()

    self.epochs = epochs

    self.num_users = len(self.user_to_index)
    self.num_movies = len(self.movie_to_index)

    self.loader = CustomLoader(data=self.df_to_index, test_size=0.1, batch_size=64)
    self.train_loader = self.loader.get_train_loader()
    self.test_loader = self.loader.get_test_loader()

    self.model = MatrixFactorization(num_users=self.num_users, num_movies=self.num_movies)

    self.trainer = Trainer(model=self.model, train_loader=self.train_loader, test_loader=self.test_loader, epochs=self.epochs)

    self.predictor = Predictor(model=self.model, user_to_index=self.user_to_index, movie_to_index=self.movie_to_index, df_to_index=self.df_to_index)

    self.index_to_user = self.predictor.index_to_user
    self.index_to_movie = self.predictor.index_to_movie

  def train(self):
    self.trainer.train()

  def test(self):
    self.trainer.test()

  def predict(self):
    """
      trả về bảng ban đầu với real index cuả user, movie
    """
    predictions_index = self.predictor.predict()
    res = np.full(shape=(max(self.user_to_index.keys())+1, max(self.movie_to_index.keys())+ 1),fill_value=3.)

    for i in range(predictions_index.shape[0]):
      for j in range(predictions_index.shape[1]):
        res[self.index_to_user[i], self.index_to_movie[j]] = predictions_index[i, j].item()

    return res

