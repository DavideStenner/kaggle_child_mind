from typing import Any, Tuple

from src.model.ensemble.initialize import EnsembleInit
from src.base.ensemble.ensemble import EnsemblePipeline
from src.model.ensemble.training import EnsembleTrainer

class EnsemblePipeline(EnsemblePipeline, EnsembleTrainer):
    def __init__(self, 
            experiment_name:str, 
            params_ensemble: dict[str, Any],
            config_dict: dict[str, Any],
        ):
        EnsembleInit.__init__(
            self, experiment_name=experiment_name, 
            params_ensemble=params_ensemble, config_dict=config_dict
        )
        self.load_model()

    def activate_inference(self) -> None:
        self.inference = True
        
    def ensemble_preprocess(self) -> None:
        self.initialize_logger()
        self.ensemble_logger.info('Starting preprocessing ensemble')
        self.train()
    
    def predict(self) -> None: 
        pass