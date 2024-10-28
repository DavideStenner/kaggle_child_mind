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
        self.__create_base_feature()
        self.__create_time_series_feature()
    
    def __create_base_feature(self) -> None:
        self.base_data = (
            self.base_data
            .with_columns(
                (pl.col('Physical-BMI') * pl.col('Basic_Demos-Age')).alias('BMI_Age'),
                (pl.col('PreInt_EduHx-computerinternet_hoursday') * pl.col('Basic_Demos-Age')).alias('Internet_Hours_Age'),
                (pl.col('PreInt_EduHx-computerinternet_hoursday') * pl.col('Physical-BMI')).alias('BMI_Internet_Hours'),
                (pl.col('BIA-BIA_Fat') / pl.col('BIA-BIA_BMI')).alias('BFP_BMI'),
                (pl.col('BIA-BIA_FFMI') / pl.col('BIA-BIA_Fat')).alias('FFMI_BFP'),
                (pl.col('BIA-BIA_FMI') / pl.col('BIA-BIA_Fat')).alias('FMI_BFP'),
                (pl.col('BIA-BIA_LST') / pl.col('BIA-BIA_TBW')).alias('LST_TBW'),
                (pl.col('BIA-BIA_Fat') * pl.col('BIA-BIA_BMR')).alias('BFP_BMR'),
                (pl.col('BIA-BIA_Fat') * pl.col('BIA-BIA_DEE')).alias('BFP_DEE'),
                (pl.col('BIA-BIA_BMR') / pl.col('Physical-Weight')).alias('BMR_Weight'),
                (pl.col('BIA-BIA_DEE') / pl.col('Physical-Weight')).alias('DEE_Weight'),
                (pl.col('BIA-BIA_SMM') / pl.col('Physical-Height')).alias('SMM_Height'),
                (pl.col('BIA-BIA_SMM') / pl.col('BIA-BIA_FMI')).alias('Muscle_to_Fat'),
                (pl.col('BIA-BIA_TBW') / pl.col('Physical-Weight')).alias('Hydration_Status'),
                (pl.col('BIA-BIA_ICW') / pl.col('BIA-BIA_TBW')).alias('ICW_TBW'),
            )
        )
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
                    .with_columns(
                        [
                            pl.sum_horizontal(
                                [
                                    f'{col}_count'
                                    for col in self.config_dict['COLUMN_INFO']['TIME_SERIES_FEATURES']
                                ]
                            ).cast(pl.UInt32).alias(f'total_count_dataset')
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
            .with_columns(
                pl.col('total_count_dataset').fill_null(0)
            )
        )