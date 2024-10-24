import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb

from tqdm import tqdm

from src.base.model.inference import ModelPredict
from src.model.lgbm.initialize import LgbmInit

class LgbmInference(ModelPredict, LgbmInit):     
    def load_feature_data(self, data: pl.DataFrame) -> np.ndarray:
        return data.select(self.feature_list).to_pandas().to_numpy(dtype=self.feature_precision)
        
    def blend_model_predict(self, test_data: pl.DataFrame, model_list: list[lgb.Booster], epoch: int) -> np.ndarray:        
        test_data = self.load_feature_data(test_data)
                
        for iter, model in enumerate(model_list):
            prediction_model = model.predict(
                test_data, num_iteration = epoch
            )/self.n_fold
            
            if iter==0:
                prediction_ = prediction_model
            else:
                prediction_ = np.add(prediction_, prediction_model)
            
        return prediction_
    
    def predict(self, test_data: pl.DataFrame) -> pl.Series:
        assert self.inference

        self.load_used_feature(model_type='main')
        self.load_used_categorical_feature(model_type='main')
        
        best_epoch = self.load_best_result(
            model_type='main'
        )['best_epoch']
        model_list: list[lgb.Booster] = self.load_pickle_model_list(
            model_type='main', 
        )

        prediction_ = self.blend_model_predict(
            test_data=test_data, model_list=model_list, epoch=best_epoch
        )
            
        return pl.Series(self.target_col, prediction_)