from abc import ABC, abstractmethod

class EnsembleInit(ABC):    
    @abstractmethod
    def load_model(self) -> None: 
        pass