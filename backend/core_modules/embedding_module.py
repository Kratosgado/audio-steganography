from abc import ABC, abstractmethod

class EmbeddingModule(ABC):
  def __init__(self):
    pass

  @abstractmethod
  def embed(self, *args, **kwargs) -> np.ndarray:
    pass

  @abstractmethod
  def extract(self, *args, **kwargs) -> list[int]:
    pass

  @abstractmethod
  def set_parameters(self, action):
    pass