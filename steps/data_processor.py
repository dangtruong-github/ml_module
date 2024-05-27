import pandas as pd 
import numpy as np 

"""
DataProcesser
"""
class DataProcesser:
  def __init__(self, df):
    """
      Arg:
      df: dataframe nguyên bản từ data base dạng (user_id, movie_id, rating)
    """
    self.df = df

  def transform_to_index(self):

    """
      chuyển những user_id, movie_id nguyên bản về  dạng index (đánh index lại từ 0)
      hàm trả về dict map (user_id_raw: user_id_new), tương tự với movie, rating để nguyên
    """
    users = self.df['user_id'].unique()
    movies = self.df['movie_id'].unique()
    user_to_index = {user_id: index for index, user_id in enumerate(users)}
    movie_to_index = {movie_id: index for index, movie_id in enumerate(movies)}

    df_to_index = []

    for _, row in self.df.iterrows():
      user_index, movie_index, rating = user_to_index[int(row['user_id'])], movie_to_index[int(row['movie_id'])], row['rating']
      df_to_index.append([user_index, movie_index, rating])

    df_to_index = pd.DataFrame(df_to_index, columns=['user_id', 'movie_id', 'rating'])
    return user_to_index, movie_to_index, df_to_index


