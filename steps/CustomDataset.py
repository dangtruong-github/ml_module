from torch.utils.data import Dataset
import numpy as np 

class CustomDataset(Dataset):
  def __init__(self, data):

    """
      Args: data: pd.DataFrame with columns: userId, movieId, rating
      đây là data frame đã được chuyển về dạng index mới và rating được scale

    """
    super().__init__()
    self.data = data

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    values = self.data.iloc[idx].values
    return dict(
        user=int(values[0]), movie=int(values[1]), rating=np.float32(values[2]/5.0)
    )
