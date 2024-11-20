from typing import Any, Tuple

from src.model.ensemble.initialize import EnsembleInit
from src.base.ensemble.ensemble import EnsemblePipeline
from src.model.ensemble.training import EnsembleTrainer
from src.model.ensemble.inference import EnsembleInference

class EnsemblePipeline(EnsemblePipeline, EnsembleTrainer, EnsembleInference):
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
        for pipeline_model in self.pipeline_model_list:
            pipeline_model.activate_inference()
        
    def ensemble_preprocess(self) -> None:
        self.initialize_logger()
        self.ensemble_logger.info('Starting preprocessing ensemble')
        self.train()