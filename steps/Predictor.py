import torch
from torch import nn 

import pandas as pd 

class Predictor:
  def __init__(
      self,
      model: nn.Module,
      user_to_index: dict,
      movie_to_index: dict,
      df_to_index: pd.DataFrame,
      device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ):

    """
      Args: model: mô hình đã được train
            user_to_index: map từ real index sang new index
            movie_to_index: map từ real index sang new index
            df_to_index: map từ real index sang new index, rating được scale
    """
    self.model = model
    self.user_to_index = user_to_index
    self.movie_to_index = movie_to_index
    self.df_to_index = df_to_index
    self.device = device

    self.num_users = len(user_to_index)
    self.num_movies = len(movie_to_index)

    self.index_to_user = {index: user_id for user_id, index in user_to_index.items()}
    self.index_to_movie = {index: movie_id for movie_id, index in movie_to_index.items()}

  def predict(self):
    """
    return matrix m(user) x n(movie)
    predict with index data not real data

    """

    self.model.to(self.device)
    self.model.eval()

    result = torch.zeros(size=(self.num_users, self.num_movies))

    for _, row in self.df_to_index.iterrows():
            user_index = int(row['user_id'])
            movie_index = int(row['movie_id'])
            rating = row['rating']
            result[user_index, movie_index] = rating

    unknown_users = []
    unknown_movies = []

    for i in range(self.num_users):
      for j in range(self.num_movies):
        if result[i, j] == 0:
          unknown_users.append(i)
          unknown_movies.append(j)

    unknown_users = torch.tensor(unknown_users, dtype=torch.int64, device=self.device)
    unknown_movies = torch.tensor(unknown_movies, dtype=torch.int64, device=self.device)


    with torch.no_grad():
        predictions = self.model(unknown_users, unknown_movies).detach().cpu()


    for idx in range(predictions.size(0)):
        result[unknown_users[idx], unknown_movies[idx]] = predictions[idx]

    return result.cpu()
