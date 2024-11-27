import numpy as np
import polars as pl
import pandas as pd
import xgboost as xgb

from src.base.model.inference import ModelPredict
from src.model.nn.initialize import MLPInit

class MLPInference(ModelPredict, MLPInit):     
    def load_feature_data(self, data: pl.DataFrame) -> pd.DataFrame:
        feature_data = data.select(self.feature_list).to_pandas()
        return feature_data
        
    def blend_model_predict(self, test_data: pl.DataFrame, model_list: list[xgb.Booster], epoch: int) -> np.ndarray:             
        prediction_ = np.zeros((test_data.shape[0]), dtype=self.target_precision)
        test_data = self.load_feature_data(test_data)
        
        for iter_, model in enumerate(model_list):
            prediction_model = model.predict(
                test_data,
                iteration_range = (0, epoch)
            )/self.n_fold

            if iter_==0:
                prediction_ = prediction_model
            else:
                prediction_ = np.add(prediction_, prediction_model)
            
        return prediction_

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
    
    def treshold_predict(self, model_type: str, test_data: pl.DataFrame) -> np.ndarray:
        prediction_ = self.predict(model_type=model_type, test_data=test_data)
        best_result = self.load_best_result(
            model_type=model_type
        )
        
        best_combination = best_result['treshold_optim']['best_combination']
        rounded_prediciton_ = self.round_predict(
            prediction_= prediction_,
            best_combination=best_combination
        )
            
        return rounded_prediciton_
    
    def predict(self, model_type: str, test_data: pl.DataFrame) -> np.ndarray:
        assert self.inference

        self.load_used_feature(model_type=model_type)
        self.load_used_categorical_feature(model_type=model_type)
        self.load_best_result(model_type=model_type)
        best_result = self.load_best_result(
            model_type=model_type
        )
        
        best_epoch = best_result['best_epoch']
        model_list: list[xgb.Booster] = self.load_pickle_model_list(
            model_type=model_type, 
        )

        prediction_= self.blend_model_predict(
            test_data=test_data, model_list=model_list, epoch=best_epoch
        )
            
        return prediction_