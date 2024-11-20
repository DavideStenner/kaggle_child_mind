from abc import ABC, abstractmethod

class EnsembleInference(ABC):    
    @abstractmethod
    def predict(self) -> None: 
        pass