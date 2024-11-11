from abc import ABC, abstractmethod

class EnsemblePipeline(ABC):
    @abstractmethod
    def predict(self) -> None: 
        pass