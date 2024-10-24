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
        id_data = (
            self.data
            .select(
                self.id_col, 
                'Basic_Demos-Sex', 'sii',
            )
            .select(
                self.id_col,
                (
                    pl.col('Basic_Demos-Sex').cast(pl.Utf8) +
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
            data=self.data, 
            fold_mapper=fold_mapper
        )
        return data

    def create_fold(self) -> None:        
        self.data: pl.DataFrame = self.__create_binary_fold()