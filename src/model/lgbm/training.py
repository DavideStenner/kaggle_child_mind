import os
import gc
import numpy as np
import polars as pl
import lightgbm as lgb

from functools import partial
from typing import Tuple, Dict

from src.base.model.training import ModelTrain
from src.model.lgbm.initialize import LgbmInit
from src.model.metric.official_metric import lgb_quadratic_kappa

class LgbmTrainer(ModelTrain, LgbmInit):
    def _init_train(self) -> None:
        data = pl.scan_parquet(
            os.path.join(
                self.config_dict['PATH_GOLD_DATA'],
                'data.parquet'
            )
        )
        
        self.feature_list = [
            col for col in data.collect_schema().names()
            if col not in self.useless_col_list + [self.target_col]
        ]
        self.categorical_col_list: list[str] = [
            col for col in self.categorical_col_list
            if 
                (col not in self.useless_col_list) & 
                (col in data.collect_schema().names())
        ]
        self.training_logger.info(f'Using {len(self.categorical_col_list)} categorical features')

    def access_fold(self, fold_: int) -> pl.LazyFrame:
        fold_data = (
            pl.scan_parquet(
                os.path.join(
                    self.config_dict['PATH_GOLD_DATA'],
                    f'data.parquet'
                )
            ).with_columns(
                (
                    pl.col('fold_info').str.split(', ')
                    .list.get(fold_).alias('current_fold')
                )
            )
        )
        return fold_data


    def train_target(self, fold_: int, model_type: str) -> None:
        #classification metric
        params_lgb = self.params_lgb
        progress = {}

        callbacks_list = [
            lgb.record_evaluation(progress),
            lgb.log_evaluation(
                period=100, 
                show_stdv=False
            )
        ]

        train_matrix, test_matrix = self.get_dataset(fold_=fold_)

        self.training_logger.info(f'Start {model_type} training')
        model = lgb.train(
            params=params_lgb,
            train_set=train_matrix, 
            num_boost_round=params_lgb['n_round'],
            valid_sets=[test_matrix],
            valid_names=['valid'],
            callbacks=callbacks_list,
            feval=lgb_quadratic_kappa
        )

        model.save_model(
            os.path.join(
                self.experiment_model_path.format(model_type=model_type),
                (
                    self.model_file_name_dict['model_list'][model_type]
                    .format(fold_=fold_)
                )
            ), importance_type='gain'
        )

        setattr(
            self, f"model_{model_type}_list",
            (
                getattr(
                    self, f"model_{model_type}_list"
                ) +
                [model]
            )
        )
        setattr(
            self, f"progress_{model_type}_list",
            (
                getattr(
                    self, f"progress_{model_type}_list"
                ) +
                [progress]
            )
        )

        del train_matrix, test_matrix
        
        _ = gc.collect()

    def get_dataset(self, fold_: int) -> Tuple[lgb.Dataset]:
        fold_data = self.access_fold(fold_=fold_)
                    
        train_filtered = fold_data.filter(
            (pl.col('current_fold') == 't')
        )
        test_filtered = fold_data.filter(
            (pl.col('current_fold') == 'v')
        )
        
        assert len(
            set(
                train_filtered.select(self.id_row).unique().collect().to_series().to_list()
            ).intersection(
                test_filtered.select(self.id_row).unique().collect().to_series().to_list()
            )
        ) == 0
                        
        train_matrix = lgb.Dataset(
            train_filtered.select(self.feature_list).collect().to_pandas().to_numpy(self.feature_precision),
            train_filtered.select(self.target_col).collect().to_pandas().to_numpy(self.target_precision),
            feature_name=self.feature_list, categorical_feature=self.categorical_col_list
        )
        test_matrix = lgb.Dataset(
            test_filtered.select(self.feature_list).collect().to_pandas().to_numpy(self.feature_precision),
            test_filtered.select(self.target_col).collect().to_pandas().to_numpy(self.target_precision),
            feature_name=self.feature_list, categorical_feature=self.categorical_col_list
        )
        return train_matrix, test_matrix

    def train(self) -> None:
        
        self._init_train()
        
        for model_type in self.model_used:
            
            #save feature list locally for later
            self.save_used_feature(model_type=model_type, feature_list=self.feature_list)
            self.save_used_categorical_feature(model_type=model_type)
            
            self.training_logger.info(f'Start {model_type} with {len(self.feature_list)} features')

            for fold_ in range(self.n_fold):
                self.training_logger.info(f'\n\nStarting fold {fold_}\n\n\n')
                self.training_logger.info('Collecting dataset')
        
                self.train_target(fold_=fold_, model_type=model_type)
            
            self.save_model(target=model_type)
            
    def save_model(self, target: str)->None:            
        self.save_pickle_model_list(
            getattr(
                self, f'model_{target}_list'
            ), 
            target,
        )
        self.save_progress_list(
            getattr(
                self, f'progress_{target}_list'
            ), 
            target
        )