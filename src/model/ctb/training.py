import os
import gc
import numpy as np
import pandas as pd
import polars as pl
import catboost as cb

from functools import partial
from typing import Tuple, Dict

from src.base.model.training import ModelTrain
from src.model.ctb.initialize import CtbInit
from src.model.metric.official_metric import QuadraticKappa

class CtbTrainer(ModelTrain, CtbInit):
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
            if col in self.feature_list
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
        train_matrix, test_matrix = self.get_dataset(fold_=fold_)

        self.training_logger.info(f'Start {model_type} training')
        model = cb.CatBoostRegressor(
            **self.params_ctb,
            allow_writing_files=False, use_best_model=False,
            eval_metric=QuadraticKappa()
        )
        model.fit(
            train_matrix, 
            eval_set=test_matrix, 
            verbose=100
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
                [model.get_evals_result()]
            )
        )

        del train_matrix, test_matrix
        
        _ = gc.collect()

    def __check_validity_train_dataset(self) -> None:
        NUMERIC_POLARS_DTYPES: list[pl.DataType] = [
            pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64, 
        ]
        STRING_POLARS_DTYPES: list[pl.DataType] = [pl.Utf8]

        data = self.access_fold(fold_=0)
        feature_dataset = data.select(self.feature_list).clone()
        
        #catboost use string for categorical....
        original_columns = feature_dataset.collect_schema().names()
        accepted_dtypes_col = feature_dataset.select(pl.all()).select(pl.col(NUMERIC_POLARS_DTYPES+STRING_POLARS_DTYPES)).collect_schema().names()
        
        #no timestamp or string
        assert set(original_columns) == set(accepted_dtypes_col)
        
        sum_infinite_value: int = (
            feature_dataset
            .select(pl.exclude(pl.Utf8).is_infinite().sum())
            .select(pl.sum_horizontal(pl.all()).alias('any_infinite'))
            .collect()
            .item()
        )
        assert sum_infinite_value == 0
        
    def get_dataset(self, fold_: int) -> Tuple[cb.Pool]:
        fold_data = self.access_fold(fold_=fold_)
                    
        train_filtered = (
            fold_data
            .filter(
                (pl.col('current_fold') == 't')
            )
            .with_columns(
                [
                    pl.col(col).cast(pl.Utf8).fill_null('none')
                    for col in self.categorical_col_list
                ]
            )
        )
            
        test_filtered = (
            fold_data
            .filter(
                (pl.col('current_fold') == 'v')
            )
            .with_columns(
                [
                    pl.col(col).cast(pl.Utf8).fill_null('none')
                    for col in self.categorical_col_list
                ]
            )
        )
        
        if not self.config_dict['ONLINE']:
            assert len(
                set(
                    train_filtered.select(self.id_row).unique().collect().to_series().to_list()
                ).intersection(
                    test_filtered.select(self.id_row).unique().collect().to_series().to_list()
                )
            ) == 0
        
        train_matrix = cb.Pool(
            data=train_filtered.select(self.feature_list).collect().to_pandas(),
            label=train_filtered.select(self.target_col).collect().to_pandas(),
            feature_names=self.feature_list, cat_features=self.categorical_col_list
        )
        test_matrix = cb.Pool(
            data=test_filtered.select(self.feature_list).collect().to_pandas(),
            label=test_filtered.select(self.target_col).collect().to_pandas(),
            feature_names=self.feature_list, cat_features=self.categorical_col_list
        )
        return train_matrix, test_matrix

    def train(self) -> None:
        
        self._init_train()
        
        for model_type in self.model_used:
            
            #save feature list locally for later
            self.save_used_feature(model_type=model_type, feature_list=self.feature_list)
            self.save_used_categorical_feature(model_type=model_type)
            self.__check_validity_train_dataset()
            
            self.training_logger.info(f'Start {model_type} with {len(self.feature_list)} features')

            for fold_ in range(self.n_fold):
                self.training_logger.info(f'\n\nStarting fold {fold_}\n\n\n')
                self.training_logger.info('Collecting dataset')
        
                self.train_target(fold_=fold_, model_type=model_type)
            
            self.save_model(target=model_type)
            
    def begin_pseudo_label(self, model_type: str) -> str:
        assert model_type in self.model_used
        self.training_logger.info(f'Starting pseudo labeling {model_type} experiment')
        
        self.load_used_feature(model_type=model_type)
        self.load_used_categorical_feature(model_type=model_type)
        
        self.save_oof_pseudo_label(model_type=model_type)
        
        if not model_type.split('_')[-1].isnumeric():
            new_model_name = f'{model_type}_pseudo_0'
        else:
            base_name: str = '_'.join(model_type.split('_')[:-1])
            pseudo_iteration: int = int(model_type.split('_')[-1])
            new_model_name = f'{base_name}_{pseudo_iteration+1}'
        
        self.model_used = [new_model_name]
        self.model_metric_used[new_model_name] = self.model_metric_used[model_type]
        
        self.initialize_model_utils()
        self.get_model_file_name_dict()
        
        return new_model_name

    def save_oof_pseudo_label(self, model_type: str) -> None:
        best_result_ = self.load_best_result(model_type=model_type)
        model_list = self.load_pickle_model_list(model_type=model_type)
        assert best_result_
        
        null_data = (
            pl.read_parquet(
                os.path.join(
                    self.original_path_gold,
                    f'data_null.parquet'
                )
            )
            .with_columns(
                [
                    pl.col(col).cast(pl.Utf8).fill_null('none')
                    for col in self.categorical_col_list
                ]
            )
        )
        test_data = (
            pl.read_parquet(
                os.path.join(
                    self.original_path_gold,
                    f'test_data.parquet'
                )
            )
            .with_columns(
                [
                    pl.col(col).cast(pl.Utf8).fill_null('none')
                    for col in self.categorical_col_list
                ]
            )
        )
        base_data_casted = (
            pl.read_parquet(
                os.path.join(
                    self.original_path_gold,
                    f'data.parquet'
                )
            )
            .with_columns(pl.col(self.target_col).cast(pl.Float64))
            .with_columns(
                [
                    pl.col(col).cast(pl.Utf8).fill_null('none')
                    for col in self.categorical_col_list
                ]
            )
        )
        list_pseudo_null_dataset: list[pl.DataFrame] = []
        
        if self.config_dict['ONLINE']:
            list_pseudo_test_dataset: list[pl.DataFrame] = []

        for fold_ in range(self.n_fold):
            fold_data = (
                pl.scan_parquet(
                    os.path.join(
                        self.original_path_gold,
                        f'data.parquet'
                    )
                ).with_columns(
                    (
                        pl.col('fold_info').str.split(', ')
                        .list.get(fold_).alias('current_fold')
                    )
                )
            )
                    
            pseudo_train_current_fold: str = (
                fold_data.filter(
                    (pl.col('current_fold') == 'v')
                )
                .select(pl.col('fold_info').first())
                .collect()
                .item()
            )
            pseudo_train_current_fold = (
                pseudo_train_current_fold
                .replace('t', 'n')
                .replace('v', 't')
            )
            pseudo_null = null_data.to_pandas().copy(deep=True)
            pseudo_test = test_data.to_pandas().copy(deep=True)
            
            oof_prediction: np.ndarray = model_list[fold_].predict(
                data=pseudo_null[self.feature_list],
                ntree_end=best_result_['best_epoch']
            )
            pseudo_null[self.target_col] = oof_prediction
            pseudo_null['fold_info'] = pseudo_train_current_fold
            
            list_pseudo_null_dataset.append(
                pl.from_dataframe(pseudo_null).with_columns(pl.col(self.target_col).cast(pl.Float64)),
            )

            if self.config_dict['ONLINE']:
                oof_prediction_test: np.ndarray = model_list[fold_].predict(
                    data=pseudo_test[self.feature_list],
                    ntree_end=best_result_['best_epoch']
                )
                pseudo_test[self.target_col] = oof_prediction_test
                pseudo_test['fold_info'] = pseudo_train_current_fold

                list_pseudo_test_dataset.append(
                    pl.from_dataframe(pseudo_test).with_columns(pl.col(self.target_col).cast(pl.Float64))
                )
            
        #concat and select common columns
        if self.config_dict['ONLINE']:
            pseudo_test_dataset = pl.concat(list_pseudo_test_dataset)
            keep_col_list: list[str] = pseudo_test_dataset.collect_schema().names()
        else:
            keep_col_list: list[str] = base_data_casted.collect_schema().names()
            
        pseudo_null_dataset = (
            pl.concat(list_pseudo_null_dataset)
            .select(keep_col_list)
        )
        base_data_casted = (
            base_data_casted
            .select(keep_col_list)
        )
        
        if self.config_dict['ONLINE']:
            pseudo_dataset = (
                pl.concat(
                    [
                        pseudo_null_dataset.cast(base_data_casted.collect_schema()), 
                        pseudo_test_dataset.cast(base_data_casted.collect_schema())
                    ]
                )
            )
        else:
            pseudo_dataset = pseudo_null_dataset.cast(base_data_casted.collect_schema())

        pseudo_dataset: pl.DataFrame = pl.concat(
            [base_data_casted, pseudo_dataset]
        )
        self.training_logger.info(f'Saving pseudo labeled dataset')
        
        (
            pseudo_dataset
            .write_parquet(
            os.path.join(
                    self.experiment_path_dict['pseudo_labeling'].format(model_type=model_type),
                    f'data.parquet'
                )
            )
        )
        self.config_dict['PATH_GOLD_DATA'] = self.experiment_path_dict['pseudo_labeling'].format(model_type=model_type)
        self.training_logger.info(f"Changed gold path to {self.config_dict['PATH_GOLD_DATA']}")
            
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