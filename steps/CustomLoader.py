from sklearn.model_selection import train_test_split 
from .CustomDataset import CustomDataset

from torch.utils.data import DataLoader 

class CustomLoader:
  def __init__(self, data, test_size, batch_size, random_state=42):

    """
      Args: data: pd.DataFrame
            test_size: float
            batch_size: int

            data: data đã được reindex và scale ratine
    """
    self.data = data
    self.test_size = test_size
    self.batch_size = batch_size
    self.random_state = random_state
    self.train_data, self.test_data = train_test_split(data, test_size=test_size, shuffle=True, random_state=self.random_state)

  def get_train_loader(self):
    """
    trả về train loader
    """
    train_dataset = CustomDataset(self.train_data)
    return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

  def get_test_loader(self):
    """"
    trả về test loader
    """
    test_dataset = CustomDataset(self.test_data)
    return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)