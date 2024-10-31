from typing import Any, Tuple

from src.model.lgbm.training import LgbmTrainer
from src.model.lgbm.explainer import LgbmExplainer
from src.model.lgbm.initialize import LgbmInit
from src.model.lgbm.inference import LgbmInference
from src.base.model.pipeline import ModelPipeline

class LgbmPipeline(ModelPipeline, LgbmTrainer, LgbmExplainer, LgbmInference):
    def __init__(self, 
            experiment_name:str, 
            params_lgb: dict[str, Any],
            config_dict: dict[str, Any],
            fold_name: str = 'fold_info', 
            evaluate_shap: bool=False
        ):
        LgbmInit.__init__(
            self, experiment_name=experiment_name, params_lgb=params_lgb,
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
        self.get_oof_insight()
        self.get_oof_prediction()
    
    def pseudo_label_train(self) -> None:   
        import numpy as np    
        model_type = 'main'
        
        pseudo_label_score_list: list[str] = [
            self.load_best_result(model_type=model_type)['treshold_optim']['best_score']
        ]
        pseudo_experiment_name_list: list[str] = [model_type]
        
        for _ in range(5):
            model_type = self.begin_pseudo_label(model_type=model_type)
            self.create_experiment_structure()
            self.run_train()
            self.explain_model()

            pseudo_experiment_name_list.append(model_type)
            pseudo_label_score_list.append(
                self.load_best_result(model_type=model_type)['treshold_optim']['best_score']
            )
        
        self.training_logger.info('\n\n')
        
        self.training_logger.info(f"{'\n'.join([str(x) for x in pseudo_label_score_list])}")
        
        best_pseudo: int = int(np.argmax(pseudo_label_score_list))
        best_pseudo_score: float = pseudo_label_score_list[best_pseudo]
        
        pseudo_result = {
            'pseudo_score': pseudo_label_score_list,
            'best_pseudo_score': best_pseudo_score,
            'best_pseudo': best_pseudo,
            'best_pseudo_name': pseudo_experiment_name_list[best_pseudo]
        }

        self.training_logger.info(f'Best pseudo labeling score: {best_pseudo_score:.6f} for a total of {best_pseudo} consecutive pseudo labeling')
        self.training_logger.info(f'Improvement of {(best_pseudo_score-pseudo_label_score_list[0]):.6f}')
        self.save_best_pseudo_result(pseudo_result=pseudo_result)
        
    def train_explain(self) -> None:
        self.create_experiment_structure()
        self.initialize_logger()
        self.run_train()
        self.explain_model()
        self.pseudo_label_train()