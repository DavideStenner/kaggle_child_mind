from typing import Any, Tuple

from src.model.xgbm.training import XgbTrainer
from src.model.xgbm.initialize import XgbInit
from src.model.xgbm.inference import XgbInference
from src.model.xgbm.explainer import XgbExplainer

from src.base.model.pipeline import ModelPipeline

class XgbPipeline(ModelPipeline, XgbTrainer, XgbExplainer, XgbInference):
    def __init__(self, 
            experiment_name:str, 
            params_xgb: dict[str, Any],
            config_dict: dict[str, Any],
            fold_name: str = 'fold_info', 
            evaluate_shap: bool=False
        ):
        XgbInit.__init__(
            self, experiment_name=experiment_name, params_xgb=params_xgb,
            config_dict=config_dict,
            fold_name=fold_name
        )
        self.evaluate_shap: bool = evaluate_shap
        
    def activate_inference(self) -> None:
        self.inference = True
        
    def run_train(self) -> None:
        self.save_params()
        self.train()
        
    def explain_model(self) -> None:
        self.evaluate_score()
        self.get_feature_importance()
        self.get_oof_prediction()
        self.get_oof_insight()
        
    def train_explain(self) -> None:
        self.create_experiment_structure()
        self.initialize_logger()
        self.run_train()
        self.explain_model()