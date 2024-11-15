import numpy as np
import pandas as pd
import polars as pl
import catboost as cb

from tqdm import tqdm

from src.base.model.inference import ModelPredict
from src.model.ctb.initialize import CtbInit

class CtbInference(ModelPredict, CtbInit):     
    def load_feature_data(self, data: pl.DataFrame) -> np.ndarray:
        return data.select(self.feature_list).to_pandas().to_numpy(dtype=self.feature_precision)
        
    def blend_model_predict(self, test_data: pl.DataFrame, model_list: list[cb.CatBoost], epoch: int) -> np.ndarray:        
        test_data = self.load_feature_data(test_data)
                
        for iter, model in enumerate(model_list):
            prediction_model = model.predict(
                test_data, ntree_end = epoch
            )/self.n_fold
            
            if iter==0:
                prediction_ = prediction_model
            else:
                prediction_ = np.add(prediction_, prediction_model)
            
        return prediction_
    
    def predict(self, model_type: str, test_data: pl.DataFrame) -> np.ndarray:
        assert self.inference

        self.load_used_feature(model_type=model_type)
        self.load_used_categorical_feature(model_type=model_type)
        self.load_best_result(model_type=model_type)
        best_result = self.load_best_result(
            model_type=model_type
        )
        
        best_epoch = best_result['best_epoch']
        best_combination = best_result['treshold_optim']['best_combination']
        model_list: list[cb.CatBoost] = self.load_pickle_model_list(
            model_type=model_type, 
        )

        prediction_ = self.blend_model_predict(
            test_data=test_data, model_list=model_list, epoch=best_epoch
        )
        
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