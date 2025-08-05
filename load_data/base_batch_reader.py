from abc import ABC,abstractmethod
class BaseBatchReader(ABC):
    def __init__(self, **kwargs) :
        pass
    @abstractmethod
    def next_batch(self, **kwargs) :
        pass
    @abstractmethod
    def close(self) :
        pass
