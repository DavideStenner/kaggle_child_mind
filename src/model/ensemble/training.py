import os


import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from tqdm import tqdm
from functools import partial
from typing import Any, Tuple, Union

from src.model.ensemble.initialize import EnsembleInit
from src.base.ensemble.training import EnsembleTraining
from src.model.metric.official_metric import quadratic_weighted_kappa_tresh

class EnsembleTrainer(EnsembleTraining, EnsembleInit):
    def __get_list_of_oof_dataset(self) -> list[Tuple[pd.DataFrame, np.ndarray]]:
        list_dataset: list[Tuple[pd.DataFrame, np.ndarray]] = []
        
        for fold_ in range(self.n_fold):
            fold_data = (
                pl.read_parquet(
                    os.path.join(
                        self.config_dict['PATH_GOLD_DATA'],
                        f'data.parquet'
                    )
                )
                .with_columns(
                    (
                        pl.col('fold_info').str.split(', ')
                        .list.get(fold_).alias('current_fold')
                    )
                )
                .filter(
                    (pl.col('current_fold') == 'v')
                )
            )
            test_feature = (
                fold_data
                .select(self.all_feature_list)
                .to_pandas()
            )
            
            test_target = (
                fold_data
                .select(self.target_col)
                .to_pandas()
                .to_numpy('int').reshape((-1))
            )
            list_dataset.append([test_feature, test_target])
        
        return list_dataset

    def oof_get_best_treshold(self) -> None:
        
        self.set_postprocess_utils()
        self.ensemble_logger.info(f'{len(self.list_treshold_value)} postprocess treshold combinations')
        
        #load every model
        model_dict = {}
        total_number_model: int = len(self.pipeline_model_list)
        
        for pipeline_model in self.pipeline_model_list:
            best_pseudo_result = pipeline_model.load_best_pseudo_result()
            model_type: str = best_pseudo_result['best_pseudo_name']
            pipeline_model.model_used = [model_type]
            
            pipeline_model.initialize_model_utils()
            pipeline_model.get_model_file_name_dict()

            
            pipeline_model.load_used_feature(model_type=model_type)
            pipeline_model.load_used_categorical_feature(model_type=model_type)

            model_list: list[Union[lgb.Booster, xgb.Booster, cb.CatBoost]] = pipeline_model.load_pickle_model_list(
                model_type=model_type, 
            )
            best_result: dict[str, any] = pipeline_model.load_best_result(model_type=model_type)
        
            model_dict[pipeline_model.name] = {
                'model_list': model_list,
                'best_epoch': best_result['best_epoch'],
                'best_score': best_pseudo_result['best_pseudo_score'],
                'feature_list': pipeline_model.feature_list,
                'categorical_col_list': pipeline_model.categorical_col_list
            }
            self.all_feature_list += pipeline_model.feature_list
            
            if pipeline_model.name == 'xgboost':
                model_dict[pipeline_model.name].update(
                    {
                        'feature_types': [
                            (
                                'c' if col in pipeline_model.categorical_col_list
                                else 'q'
                            )
                            for col in pipeline_model.feature_list
                        ]
                    }
                )

        self.all_feature_list = list(set(self.all_feature_list))
        
        result_by_combination: np.ndarray = np.zeros(len(self.list_treshold_value))
        self.ensemble_logger.info(f'Starting ensemble treshold postprocess')

        dataset_list: list[Tuple[pd.DataFrame, pd.DataFrame]] = self.__get_list_of_oof_dataset()

        for fold_, (test_feature, test_target) in enumerate(dataset_list):            
            for n_model, (name_model, model_info) in enumerate(model_dict.items()):
                
                selected_model_list: list[Union[xgb.Booster, lgb.Booster, cb.CatBoost]] = model_info['model_list']
                best_epoch: int = model_info['best_epoch']
                feature_list: list[str] = model_info['feature_list']

                if name_model == 'xgboost':
                    feature_types: list[str] = model_info['feature_types']
                    
                    pred_target = selected_model_list[fold_].predict(
                        xgb.DMatrix(
                            data=(
                                test_feature[feature_list]
                                .to_numpy('float64')
                            ), 
                            feature_names=feature_list, 
                            enable_categorical=True, 
                            feature_types=feature_types
                        ),
                        iteration_range = (0, best_epoch)
                    )
                    
                if name_model == 'lightgbm':
                    pred_target = selected_model_list[fold_].predict(
                        (
                            test_feature[feature_list]
                            .to_numpy('float64')
                        ),
                        num_iteration = best_epoch
                    )
                    
                if name_model == 'catboost':
                    catboost_test_feature = test_feature.copy()[feature_list]
                    catboost_test_feature[model_info['categorical_col_list']] = (
                        catboost_test_feature[model_info['categorical_col_list']]
                        .fillna(-1)
                        .astype(int)
                        .astype(str)
                    )
                    catboost_test_feature[model_info['categorical_col_list']] = (
                        catboost_test_feature[model_info['categorical_col_list']]
                        .replace('-1', 'none')
                    )

                    pred_target = selected_model_list[fold_].predict(
                        catboost_test_feature[feature_list],
                        ntree_end = best_epoch
                    )
                if n_model == 0:
                    pred_target_ensemble = pred_target/total_number_model
                else:
                    pred_target_ensemble = np.add(pred_target_ensemble, pred_target/total_number_model)

            treshold_iterator = tqdm(self.list_treshold_value, total=len(self.list_treshold_value))
            
            #partial for efficiency
            partial_kappa = partial(
                quadratic_weighted_kappa_tresh,
                y_true=test_target,
                y_pred=pred_target_ensemble
            )
            result_fold = np.array(
                [
                    partial_kappa(
                        treshold_combination_
                    )/self.n_fold
                    for treshold_combination_ in treshold_iterator
                ]
            )
            result_by_combination += result_fold
        
        #find best
        idx_best_combination = np.argmax(result_by_combination)
        
        best_score = result_by_combination[idx_best_combination]
        best_combination = self.list_treshold_value[idx_best_combination]
        
        self.ensemble_logger.info(
            'Starting score:'
        )
        for name_model, model_info in model_dict.items():
            best_selected_score: float = model_info['best_score']
            self.ensemble_logger.info(
                f"{name_model}: {best_selected_score}"
            )

        self.ensemble_logger.info(
            f'Best treshold combination for ensemble with optimized score of {best_score:.6f} '
        )
        self.ensemble_logger.info(
            f'T1: {best_combination[0]}\nT2: {best_combination[1]}\nT3: {best_combination[2]}'
        )
        ensemble_best_result: dict[str, any] = {
            'best_score': best_score,
            'best_combination': best_combination
        }
        
        self.save_best_result(
            best_result=ensemble_best_result 
        )

    def train(self) -> None:
        self.oof_get_best_treshold()
    
