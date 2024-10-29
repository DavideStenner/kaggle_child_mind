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
                pl.coalesce(['PAQ_C-PAQ_C_Total', 'PAQ_A-PAQ_A_Total']).alias('PAQ_Total')
            )
            .with_columns(
                ##Demographics
                [
                    (pl.col('Physical-BMI') * pl.col('Basic_Demos-Age')).alias('new_BMI_Age'),
                    (pl.col('PreInt_EduHx-computerinternet_hoursday') * pl.col('Basic_Demos-Age')).alias('new_Internet_Hours_Age'),
                ] +
                ##Children's Global Assessment Scale
                [
                    (pl.col('Physical-BMI')/pl.col('CGAS-CGAS_Score')).alias('mine_bmi_CGAS'),
                    (pl.col('Physical-Height')/pl.col('CGAS-CGAS_Score')).alias('mine_height_CGAS'),
                    (pl.col('Physical-Weight')/pl.col('CGAS-CGAS_Score')).alias('mine_weight_CGAS'),
                    (pl.col('PAQ_Total')/pl.col('CGAS-CGAS_Score')).alias('mine_paq_total_CGAS'),
                    (pl.col('SDS-SDS_Total_T')/pl.col('CGAS-CGAS_Score')).alias('mine_sds_total_CGAS'),
                ] +
                ##Physical Measures
                [
                    (pl.col('Physical-BMI') / pl.col('PreInt_EduHx-computerinternet_hoursday')).alias('new_BMI_Internet_Hours'),
                    (pl.col('Physical-Height') / pl.col('PreInt_EduHx-computerinternet_hoursday')).alias('new_height_Internet_Hours'),
                    (pl.col('Physical-Weight') / pl.col('PreInt_EduHx-computerinternet_hoursday')).alias('new_weight_Internet_Hours'),
                    (pl.col('Physical-Systolic_BP')-pl.col('Physical-Diastolic_BP')).alias('claude_Pulse_Pressure'),
                    (
                        (pl.col('Physical-Diastolic_BP'))/
                        ((pl.col('Physical-Systolic_BP')-pl.col('Physical-Diastolic_BP'))/3)
                    ).alias('claude_Mean_Arterial_Pressure')
                ] +
                [
                    (pl.col('Physical-BMI')/pl.col(col)).alias(f'mine_bmi_{col}')
                    for col in [
                        'BIA-BIA_BMC', 'BIA-BIA_BMI', 
                        'BIA-BIA_BMR', 'BIA-BIA_DEE', 
                        'BIA-BIA_ECW', 'BIA-BIA_FFM',
                        'BIA-BIA_FFMI', 'BIA-BIA_FMI',
                        'BIA-BIA_Frame_num', 'BIA-BIA_ICW',
                        'BIA-BIA_LDM', 'BIA-BIA_LST', 'BIA-BIA_SMM',
                        'BIA-BIA_TBW'
                    ]
                ] +
                [
                    (pl.col(col)/pl.col('Physical-Height')).alias(f'mine_height_{col}')
                    for col in [
                        'BIA-BIA_BMC', 'BIA-BIA_BMI', 
                        'BIA-BIA_BMR', 'BIA-BIA_DEE', 
                        'BIA-BIA_ECW', 'BIA-BIA_FFM',
                        'BIA-BIA_FFMI', 'BIA-BIA_FMI',
                        'BIA-BIA_Frame_num', 'BIA-BIA_ICW',
                        'BIA-BIA_LDM', 'BIA-BIA_LST', 'BIA-BIA_SMM',
                        'BIA-BIA_TBW'
                    ]
                ] +
                [
                    (pl.col(col)/pl.col('Physical-Weight')).alias(f'mine_weight_{col}')
                    for col in [
                        'BIA-BIA_BMC', 'BIA-BIA_BMI', 
                        'BIA-BIA_BMR', 'BIA-BIA_DEE', 
                        'BIA-BIA_ECW', 'BIA-BIA_FFM',
                        'BIA-BIA_FFMI', 'BIA-BIA_FMI',
                        'BIA-BIA_Frame_num', 'BIA-BIA_ICW',
                        'BIA-BIA_LDM', 'BIA-BIA_LST', 'BIA-BIA_SMM',
                        'BIA-BIA_TBW'
                    ]
                ] +
                
                ##FitnessGram Vitals and Treadmill
                [
                    (pl.col('Fitness_Endurance-Time_Mins')*60+pl.col('Fitness_Endurance-Time_Sec')).alias('mine_Fitness_Endurance-Time_Total'),
                    (pl.col('Fitness_Endurance-Max_Stage')/pl.col('PreInt_EduHx-computerinternet_hoursday')).alias('mine_Fitness_Endurance_internet'),
                ] +
                
                ##FitnessGram Child
                [
                    (pl.col('FGC-FGC_SRR') + pl.col('FGC-FGC_SRL')).alias('new_SRTotal'),
                    (
                        pl.mean_horizontal(
                            [
                                'FGC-FGC_PU', 'FGC-FGC_CU', 'FGC-FGC_GSD', 
                                'FGC-FGC_GSND', 'FGC-FGC_SRL', 'FGC-FGC_SRR', 'FGC-FGC_TL'
                            ]
                        ).alias('new_horizontal_mean_FGC')
                    )
                ] +
                [
                    (
                        (pl.col('FGC-FGC_CU_Zone')) +
                        (pl.col('FGC-FGC_PU_Zone')) +
                        (pl.col('FGC-FGC_SRL_Zone')) +
                        (pl.col('FGC-FGC_SRR_Zone')) +
                        (pl.col('FGC-FGC_TL_Zone'))
                    ).cast(pl.UInt8).alias('claude_Fitness_Level'),
                    (
                        (pl.col('FGC-FGC_GSD_Zone')>=2) +
                        (pl.col('FGC-FGC_GSD_Zone')>=2) +
                        (pl.col('FGC-FGC_CU_Zone')) +
                        (pl.col('FGC-FGC_PU_Zone')) +
                        (pl.col('FGC-FGC_SRL_Zone')) +
                        (pl.col('FGC-FGC_SRR_Zone')) +
                        (pl.col('FGC-FGC_TL_Zone'))
                    ).cast(pl.UInt8).alias('claude_Fitness_Achievements'),
                    (
                        (
                            (pl.col('FGC-FGC_PU')-pl.col('FGC-FGC_PU').mean())/pl.col('FGC-FGC_PU').std() +
                            (pl.col('FGC-FGC_CU')-pl.col('FGC-FGC_CU').mean())/pl.col('FGC-FGC_CU').std() +
                            (pl.col('FGC-FGC_GSD')-pl.col('FGC-FGC_GSD').mean())/pl.col('FGC-FGC_GSD').std() +
                            (pl.col('FGC-FGC_GSND')-pl.col('FGC-FGC_GSND').mean())/pl.col('FGC-FGC_GSND').std()
                        )/4
                    ).alias('claude_Overall_Strength_Score'),
                    (
                        (
                            (pl.col('FGC-FGC_PU')-pl.col('FGC-FGC_PU').mean())/pl.col('FGC-FGC_PU').std() +
                            (pl.col('FGC-FGC_CU')-pl.col('FGC-FGC_CU').mean())/pl.col('FGC-FGC_CU').std() +
                            (pl.col('FGC-FGC_GSD')-pl.col('FGC-FGC_GSD').mean())/pl.col('FGC-FGC_GSD').std() +
                            (pl.col('FGC-FGC_GSND')-pl.col('FGC-FGC_GSND').mean())/pl.col('FGC-FGC_GSND').std() +
                            (pl.col('FGC-FGC_SRL')-pl.col('FGC-FGC_SRL').mean())/pl.col('FGC-FGC_SRL').std() +
                            (pl.col('FGC-FGC_SRR')-pl.col('FGC-FGC_SRR').mean())/pl.col('FGC-FGC_SRR').std() +
                            (pl.col('FGC-FGC_TL')-pl.col('FGC-FGC_TL').mean())/pl.col('FGC-FGC_TL').std()
                        )/4
                    ).alias('claude_Overall_FGC_score'),
                    (pl.col('FGC-FGC_SRR') + pl.col('FGC-FGC_SRL') + pl.col('FGC-FGC_TL')).alias('cluade_Overall_Flexibility_Score'),
                ] +
                
                ##Bio-electric Impedance Analysis
                [
                    (pl.col('BIA-BIA_LST') / pl.col('BIA-BIA_TBW')).alias('new_LST_TBW'),
                    (pl.col('BIA-BIA_Fat') * pl.col('BIA-BIA_BMR')).alias('new_BFP_BMR'),
                    (pl.col('BIA-BIA_Fat') * pl.col('BIA-BIA_DEE')).alias('new_BFP_DEE'),
                    (pl.col('BIA-BIA_SMM') / pl.col('BIA-BIA_FMI')).alias('new_Muscle_to_Fat'),
                    (pl.col('BIA-BIA_ICW') / pl.col('BIA-BIA_TBW')).alias('new_ICW_TBW'),
                    
                    (pl.col('BIA-BIA_ECW') / pl.col('BIA-BIA_ICW')).alias('claude_Water_Balance_Ratio'),
                    (pl.col('BIA-BIA_ECW') / pl.col('BIA-BIA_ICW')).alias('Water_Balance_Ratio'),
                ] +
                [
                    (pl.col('BIA-BIA_Fat')/pl.col(col)).alias(f'mine_fat_{col}')
                    for col in [
                        'BIA-BIA_BMC', 'BIA-BIA_BMI', 
                        'BIA-BIA_BMR', 'BIA-BIA_DEE', 
                        'BIA-BIA_ECW', 'BIA-BIA_FFM',
                        'BIA-BIA_FFMI', 'BIA-BIA_FMI',
                        'BIA-BIA_Frame_num', 'BIA-BIA_ICW',
                        'BIA-BIA_LDM', 'BIA-BIA_LST', 'BIA-BIA_SMM',
                        'BIA-BIA_TBW'
                    ]
                ] +
                ## Physical Activity Questionnaire - Sleep Disturbance Scale
                [
                    (
                        (
                            (pl.col('PAQ_Total')-pl.col('PAQ_Total').min())/
                            (pl.col('PAQ_Total').max()-pl.col('PAQ_Total').min())
                        ) +
                        (
                            1 -
                            (pl.col('SDS-SDS_Total_T')-pl.col('SDS-SDS_Total_T').min())/
                            (pl.col('SDS-SDS_Total_T').max()-pl.col('SDS-SDS_Total_T').min())
                        )
                    ).alias('claude_activit_sleep_interaction')        
                ]        
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