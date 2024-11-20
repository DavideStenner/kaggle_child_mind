from abc import ABC, abstractmethod

class EnsemblePipeline(ABC):
    @abstractmethod
    def ensemble_preprocess(self) -> None: 
        pass