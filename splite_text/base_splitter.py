from abc import ABC,abstractmethod
class BaseSplitter(ABC):
    def __init__(self,**kwargs):
        pass
    @abstractmethod
    def split(self,**kwargs):
        pass