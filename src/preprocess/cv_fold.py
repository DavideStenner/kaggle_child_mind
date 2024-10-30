import warnings
import numpy as np
import pandas as pd
import polars as pl


from tqdm import tqdm
from typing import Dict
from sklearn.model_selection import StratifiedKFold, GroupKFold, StratifiedGroupKFold
from src.base.preprocess.cv_fold import BaseCVFold
from src.preprocess.initialize import PreprocessInit

class PreprocessFoldCreator(BaseCVFold, PreprocessInit):       
    def __create_fold_from_mapper(
            self, 
            data: pl.DataFrame, fold_mapper: Dict[int, int]
        ) -> pl.DataFrame:
        data = (
            data
            .with_columns(
                pl.col(self.id_col).replace(fold_mapper).alias('fold').cast(pl.UInt8)
            )
            .with_columns(
                (
                    (
                        pl.when(
                            pl.col('fold') != fold_
                        )
                        .then(pl.lit('t'))
                        .when(
                            pl.col('fold') == fold_
                        ).then(pl.lit('v')).otherwise(pl.lit('n'))
                        .alias(f'fold_{fold_}')
                    )
                    for fold_ in range(self.n_folds)
                )
            )
            .with_columns(
                pl.concat_str(
                    [f'fold_{x}' for x in range(self.n_folds)],
                    separator=', '
                )
                .alias('fold_info')
            )
            .drop(['fold'] + [f'fold_{x}' for x in range(self.n_folds)])
        )
        return data
    
    def __create_binary_fold(self) -> pl.LazyFrame:
        splitter_ = StratifiedKFold(self.n_folds, shuffle=True)
        base_data = self.data.filter(pl.col(self.target).is_not_null())
        id_data = (
            base_data
            .with_columns(
                (
                    pl.when(pl.col('Basic_Demos-Age')<=7).then(0)
                    .when(pl.col('Basic_Demos-Age')<=8).then(1)
                    .when(pl.col('Basic_Demos-Age')<=9).then(2)
                    .when(pl.col('Basic_Demos-Age')<=10).then(3)
                    .when(pl.col('Basic_Demos-Age')<=12).then(4)
                    .when(pl.col('Basic_Demos-Age')<=14).then(5)
                    .otherwise(6)
                    .alias('Basic_Demos-Age')
                )
            )
            .select(
                self.id_col, 
                'Basic_Demos-Age', 'sii',
            )
            .select(
                self.id_col,
                (
                    pl.col('Basic_Demos-Age').cast(pl.Utf8) +
                    pl.col('sii').cast(pl.Utf8)
                ).alias('slice')
            )
            .to_pandas()
        )

        fold_iterator = enumerate(splitter_.split(id_data, y = id_data['slice']))
        fold_mapper: Dict[int, int] = {}
        
        for fold_, (_, test_index) in fold_iterator:
            fold_mapper.update(
                {
                    id_: fold_
                    for id_ in id_data.loc[test_index, self.id_col].tolist()
                }
            )
            
        data  = self.__create_fold_from_mapper(
            data=base_data, 
            fold_mapper=fold_mapper
        )
        self.preprocess_logger.info(
            '\n\n' +
            (
                data
                .group_by('sii', 'fold_info')
                .agg(pl.len())
                .sort('sii')
                .to_pandas()
                .to_markdown()
            ) +
            '\n\n' +
            (
                data
                .group_by('Basic_Demos-Sex', 'fold_info')
                .agg(pl.len())
                .sort('Basic_Demos-Sex')
                .to_pandas()
                .to_markdown()
            ) +
            '\n\n'
            
        )
        return data

    def create_fold(self) -> None:        
        self.null_data: pl.DataFrame = self.data.filter(pl.col(self.target).is_null())
        self.data: pl.DataFrame = self.__create_binary_fold()
