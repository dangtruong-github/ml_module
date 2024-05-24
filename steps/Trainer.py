import torch 
from torch import nn 
from torch.utils.data import DataLoader

import numpy as np 

from sklearn.metrics import mean_absolute_error 

class Trainer:
  def __init__(
      self,
      model: nn.Module,
      train_loader: DataLoader,
      test_loader: DataLoader,
      epochs: int,
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
      criterion=nn.MSELoss(),
  ):
    """
    model: mô hình cần học
    train_loader: train_loader
    test_loader: test_loader
    epochs: số lượng epochs

    """
    self.model = model
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.epochs = epochs
    self.device = device
    self.criterion = criterion
    self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

  def train(self):
    self.model.to(self.device)
    self.model.train()
    for epoch in range(self.epochs):
      reporting_step = 100
      for i, batch in enumerate(self.train_loader):
        users, movies, rating = batch['user'].to(self.device), batch['movie'].to(self.device), batch['rating'].to(self.device)

        output = self.model(users, movies)
        loss = self.criterion(output, rating)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if i % reporting_step == 0:
          print(f"Epoch {epoch}, step {i}, loss: {loss.item()}")

  def test(self):
    self.model.to(self.device)
    self.model.eval()
    losses = []
    for i, batch in enumerate(self.test_loader):
      users, movies, rating = batch['user'].to(self.device), batch['movie'].to(self.device), batch['rating'].to(self.device)

      output = self.model(users, movies)

      losses.append(mean_absolute_error(output.detach().cpu(), rating.detach().cpu()))

    print(f"Average MAE on test set: {np.mean(losses)}" )