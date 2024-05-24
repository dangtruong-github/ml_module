import torch
from torch import nn 

class MatrixFactorization(nn.Module):
  def __init__(self, num_users, num_movies, d_model=128, drop_out=0.1):
    """
      Args: num_users: số lượng useser đã rate
            num_movies: số lượng phim đã được rate
            d_model: embedding dimension
    """
    super(MatrixFactorization, self).__init__()
    self.num_users = num_users
    self.num_movies = num_movies
    self.d_model = d_model
    self.drop_out = drop_out

    self.embedder_user = nn.Embedding(num_users, d_model)
    self.embedder_movie = nn.Embedding(num_movies, d_model)

    self.linear_net = nn.Sequential(
        nn.BatchNorm1d(256),
        nn.Linear(256, 100),
        nn.ReLU(),
        nn.BatchNorm1d(100),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.BatchNorm1d(100),
        nn.Dropout(self.drop_out),
        nn.Linear(100, 1),
        nn.ReLU(inplace=True)
    )

  def forward(self, users, movies):
    """
    Args: user đã được reindex  (B,)
          movie đã được reindex (B,)
    """
    users = self.embedder_user(users) #(B, 128)
    movies = self.embedder_movie(movies) #(B, 128)
    x = torch.cat((users, movies), dim=1) #(B, 256)
    out = self.linear_net(x)  # (B, 1)
    out = out.squeeze(1)  # (B,)
    return out
