from abc import ABC, abstractmethod

class ModelInit(ABC):
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def create_experiment_structure(self) -> None:
        pass
    
    @abstractmethod
    def load_model(self) -> None: 
        pass