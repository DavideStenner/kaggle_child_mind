from abc import ABC, abstractmethod

class EnsembleTraining(ABC):
    @abstractmethod
    def train(self) -> None:
        pass