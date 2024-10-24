import os
import gc
import pandas as pd
import polars as pl

from tqdm import tqdm
from itertools import product
from typing import cast

from src.base.preprocess.add_feature import BaseFeature
from src.preprocess.initialize import PreprocessInit

class PreprocessAddFeature(BaseFeature, PreprocessInit):

    def create_feature(self) -> None:   
        self.__create_time_series_feature()
    
    def __create_time_series_feature(self) -> None:
        self.time_series_list: list[pl.DataFrame] = [
                (
                    pl_dataframe
                    .group_by(self.config_dict['ID_COL'])
                    .agg(
                        [
                            pl_expr(col).alias(f'{col}_{pl_expr.__name__}')
                            for pl_expr, col in product(
                                [pl.count, pl.mean, pl.std, pl.min, pl.max],
                                self.config_dict['COLUMN_INFO']['TIME_SERIES_FEATURES']
                            )
                        ] +
                        [
                            pl.quantile(col, quantile_).alias(f'{col}_quantile_{quantile_}')
                            for quantile_, col in product(
                                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                self.config_dict['COLUMN_INFO']['TIME_SERIES_FEATURES'] 
                            )
                        ]
                    )
                    .collect()
                )
                for pl_dataframe in tqdm(self.time_series_list, total=len(self.time_series_list)) 
            ]
        
        self.time_series: pl.LazyFrame = pl.concat(self.time_series_list).lazy()
        del self.time_series_list

        _ = gc.collect()
        
    def merge_all(self) -> None:
        self.data = (
            self.base_data
            .join(
                self.time_series,
                on=self.config_dict['ID_COL'], how='left'
            )
        )