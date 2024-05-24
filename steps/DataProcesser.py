import pandas as pd 
import numpy as np 

"""
DataProcesser
"""
class DataProcesser:
  def __init__(self, df):
    """
      Arg:
      df: ma trận rating nguyên bản từ database
    """
    self.df = df

  def transform_data(self):
    """
      chỉ lấy ra những cặp (user, movie) khác zero và nan  ở data base
      hàm trả về pd.DataFrame (user_id, movie_id, rating) (user_id, movie_id đều nguyên bản), rating được scale về range(0, 1)
    """

    data = []
    for user in range(self.df.shape[0]):
      for item in range(self.df.shape[1]):
        if self.df[user, item] and ~np.isnan(self.df[user, item]):
          data.append([user, item, self.df[user, item]/5.0])
    return pd.DataFrame(data, columns=["userId", "movieId", "rating"])

  def transform_to_index(self):

    """
      chuyển những user_id, movie_id nguyên bản về  dạng index (đánh index lại từ 0)
      hàm trả về dict map (user_id_raw: user_id_new), tương tự với movie, rating để nguyên
    """

    self.df = self.transform_data()
    users = self.df['userId'].unique()
    movies = self.df['movieId'].unique()
    user_to_index = {user_id: index for index, user_id in enumerate(users)}
    movie_to_index = {movie_id: index for index, movie_id in enumerate(movies)}

    df_to_index = []

    for _, row in self.df.iterrows():
      user_index, movie_index, rating = user_to_index[int(row['userId'])], movie_to_index[int(row['movieId'])], row['rating']
      df_to_index.append([user_index, movie_index, rating])

    df_to_index = pd.DataFrame(df_to_index, columns=['userId', 'movieId', 'rating'])
    return user_to_index, movie_to_index, df_to_index


