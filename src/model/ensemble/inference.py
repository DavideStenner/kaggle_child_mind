import numpy as np
import polars as pl
from typing import Any, Tuple

from src.model.ensemble.initialize import EnsembleInit
from src.base.ensemble.inference import EnsembleInference

class EnsembleInference(EnsembleInit, EnsembleInference):
    def round_predict(self, prediction_: np.ndarray, best_combination: list[float]) -> np.ndarray:
        rounded_prediciton_ = (
            np.where(
                prediction_ < best_combination[0], 0,
                np.where(
                    prediction_ < best_combination[1], 1,
                        np.where(
                            prediction_ < best_combination[2], 2, 
                            3
                        )
                    )
            )
        )
        return rounded_prediciton_

    def treshold_predict(self, test_data: pl.DataFrame) -> np.ndarray:
        prediction_ = self.predict(test_data=test_data)
        best_combination = self.load_best_result()['best_combination']
        
        rounded_prediciton_ = self.round_predict(
            prediction_= prediction_,
            best_combination=best_combination
        )
            
        return rounded_prediciton_

    def predict(self, test_data: pl.DataFrame) -> np.ndarray:
        assert self.inference
        
        best_result = self.load_best_result()
        number_model: int = len(self.pipeline_model_list)
        
        for iter_, pipeline_model in enumerate(self.pipeline_model_list):
            best_result = pipeline_model.load_best_pseudo_result()
            best_pseudo = best_result['best_pseudo_name']
            
            pipeline_model.model_used = [best_pseudo]
            
            pipeline_model.initialize_model_utils()
            pipeline_model.get_model_file_name_dict()

            pipeline_model.load_used_feature(model_type=best_pseudo)
            pipeline_model.load_used_categorical_feature(model_type=best_pseudo)
            
            prediction_model = (
                pipeline_model.predict(model_type=best_pseudo, test_data=test_data)/
                number_model
            )
            
            if iter_==0:
                prediction_ = prediction_model
            else:
                prediction_ = np.add(prediction_, prediction_model)
        
        return prediction_