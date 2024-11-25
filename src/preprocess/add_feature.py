import gc
import polars as pl

from tqdm import tqdm
from itertools import product
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
                pl.coalesce(['PAQ_C-PAQ_C_Total', 'PAQ_A-PAQ_A_Total']).alias('PAQ_Total'),
                pl.sum_horizontal(
                    pl.col(self.config_dict['COLUMN_INFO']['FEATURE_LIST']).is_null()
                ).alias('mine_feature_missing')
            )
            .with_columns(
                ##Demographics
                [
                    (pl.col('Physical-BMI') * pl.col('Basic_Demos-Age')).alias('demo_Age_bmi'),
                    (pl.col('PreInt_EduHx-computerinternet_hoursday') * pl.col('Basic_Demos-Age')).alias('demo_age_Internet_Hours'),
                    (pl.col('PreInt_EduHx-computerinternet_hoursday') / pl.col('Basic_Demos-Age')).alias('demo_age_Internet_Hours_2'),
                ] +
                ##Children's Global Assessment Scale
                [
                    (pl.col('Physical-BMI')/pl.col('CGAS-CGAS_Score')).alias('cgas_bmi'),
                    (pl.col('Physical-Height')/pl.col('CGAS-CGAS_Score')).alias('cgas_height'),
                    (pl.col('Physical-Weight')/pl.col('CGAS-CGAS_Score')).alias('cgas_weight'),
                    (pl.col('PAQ_Total')/pl.col('CGAS-CGAS_Score')).alias('cgas_paq_total_1'),
                    (pl.col('SDS-SDS_Total_T')/pl.col('CGAS-CGAS_Score')).alias('cgas_sds_total_1'),
                    (pl.col('CGAS-CGAS_Score')/(1 + pl.col('PreInt_EduHx-computerinternet_hoursday'))).alias('cgas_intenet_1'),
                    (pl.col('PAQ_Total')*pl.col('CGAS-CGAS_Score')).alias('cgas_paq_total_2'),
                    (pl.col('SDS-SDS_Total_T')*pl.col('CGAS-CGAS_Score')).alias('cgas_sds_total_2'),
                    (pl.col('CGAS-CGAS_Score')*pl.col('PreInt_EduHx-computerinternet_hoursday')).alias('cgas_intenet_2'),                ] +
                ##Physical Measures
                [
                    (pl.col('Physical-BMI') / (1 + pl.col('PreInt_EduHx-computerinternet_hoursday'))).alias('phisical_BMI_Internet_Hours'),
                    (pl.col('Physical-Height') / (1 + pl.col('PreInt_EduHx-computerinternet_hoursday'))).alias('phisical_height_Internet_Hours'),
                    (pl.col('Physical-Weight') / (1 + pl.col('PreInt_EduHx-computerinternet_hoursday'))).alias('phisical_weight_Internet_Hours'),
                    (pl.col('Physical-Systolic_BP')-pl.col('Physical-Diastolic_BP')).alias('phisical_Pulse_Pressure'),
                    (
                        pl.when(
                            pl.col('Physical-Systolic_BP') != pl.col('Physical-Diastolic_BP')
                        )
                        .then(
                            (pl.col('Physical-Diastolic_BP'))/
                            ((pl.col('Physical-Systolic_BP')-pl.col('Physical-Diastolic_BP'))/3)
                        )
                        .otherwise(None)
                    ).alias('phisical_Mean_Arterial_Pressure')
                ] +                
                ##FitnessGram Vitals and Treadmill
                [
                    (pl.col('Fitness_Endurance-Time_Mins')*60+pl.col('Fitness_Endurance-Time_Sec')).alias('fitnessgram_Fitness_Endurance-Time_Total'),
                    (pl.col('Fitness_Endurance-Max_Stage')/(1 + pl.col('PreInt_EduHx-computerinternet_hoursday'))).alias('fitnessgram_Fitness_Endurance_internet'),
                ] +
                
                ##FitnessGram Child
                [
                    (pl.col('FGC-FGC_SRR') + pl.col('FGC-FGC_SRL')).alias('fitnessgramchild_SRTotal'),
                    (
                        pl.mean_horizontal(
                            [
                                'FGC-FGC_PU', 'FGC-FGC_CU', 'FGC-FGC_GSD', 
                                'FGC-FGC_GSND', 'FGC-FGC_SRL', 'FGC-FGC_SRR', 'FGC-FGC_TL'
                            ]
                        ).alias('fitnessgramchild_horizontal_mean_FGC')
                    )
                ] +
                [
                    (
                        (pl.col('FGC-FGC_CU_Zone')) +
                        (pl.col('FGC-FGC_PU_Zone')) +
                        (pl.col('FGC-FGC_SRL_Zone')) +
                        (pl.col('FGC-FGC_SRR_Zone')) +
                        (pl.col('FGC-FGC_TL_Zone'))
                    ).cast(pl.UInt8).alias('fitnessgramchild_Fitness_Level'),
                    (
                        (pl.col('FGC-FGC_GSD_Zone')>=2) +
                        (pl.col('FGC-FGC_GSD_Zone')>=2) +
                        (pl.col('FGC-FGC_CU_Zone')) +
                        (pl.col('FGC-FGC_PU_Zone')) +
                        (pl.col('FGC-FGC_SRL_Zone')) +
                        (pl.col('FGC-FGC_SRR_Zone')) +
                        (pl.col('FGC-FGC_TL_Zone'))
                    ).cast(pl.UInt8).alias('fitnessgramchild_Fitness_Achievements'),
                    (pl.col('FGC-FGC_SRR') + pl.col('FGC-FGC_SRL') + pl.col('FGC-FGC_TL')).alias('fitnessgramchild_Overall_Flexibility_Score'),
                ] +
                
                ##Bio-electric Impedance Analysis
                [
                    (pl.col('BIA-BIA_Fat') / (1 + pl.col('BIA-BIA_BMI'))).alias('bio_BFP_BMI'),
                    (pl.col('BIA-BIA_FFMI') / (1 + pl.col('BIA-BIA_Fat'))).alias('bio_FFMI_BFP'),
                    (pl.col('BIA-BIA_FMI') / (1 + pl.col('BIA-BIA_Fat'))).alias('bio_FMI_BFP'),
                    (pl.col('BIA-BIA_LST') / (1 + pl.col('BIA-BIA_TBW'))).alias('bio_LST_TBW'),
                    (pl.col('BIA-BIA_Fat') * pl.col('BIA-BIA_BMR')).alias('bio_BFP_BMR'),
                    (pl.col('BIA-BIA_Fat') * pl.col('BIA-BIA_DEE')).alias('bio_BFP_DEE'),
                    (
                        pl.when(
                            pl.col('BIA-BIA_FMI') == 0
                        )
                        .then(None)
                        .otherwise(
                         pl.col('BIA-BIA_SMM') / pl.col('BIA-BIA_FMI')
                        )
                        .alias('bio_Muscle_to_Fat')
                    ),
                    (pl.col('BIA-BIA_ICW') / (1 + pl.col('BIA-BIA_TBW'))).alias('bio_ICW_TBW'),
                    
                    (pl.col('BIA-BIA_ECW') / (1 + pl.col('BIA-BIA_ICW'))).alias('bio_Water_Balance_Ratio'),
                    (pl.col('BIA-BIA_BMR') / (1 + pl.col('Physical-Weight'))).alias('bio_BMR_Weight'),
                    
                    (pl.col('BIA-BIA_DEE') / (1 + pl.col('Physical-Weight'))).alias('bio_DEE_Weight'),
                    (pl.col('BIA-BIA_SMM') / (1 + pl.col('Physical-Height'))).alias('bio_SMM_Height'),
                    (pl.col('BIA-BIA_TBW') / (1 + pl.col('Physical-Weight'))).alias('bio_Hydration_Status'),
                ] +
                ## Physical Activity Questionnaire - Sleep Disturbance Scale
                [
                    (
                       pl.col('PAQ_Total')/(1 + pl.col('SDS-SDS_Total_T'))
                    ).alias('sleep_activit_sleep_interaction_1'),
                    (
                       pl.col('PAQ_Total')*pl.col('SDS-SDS_Total_T')
                    ).alias('sleep_activit_sleep_interaction_2')        
                ] +
                ## Internet Use
                [
                   (pl.col('PreInt_EduHx-computerinternet_hoursday')*pl.col('SDS-SDS_Total_T')).alias('internet_sds_1'), 
                   (pl.col('PreInt_EduHx-computerinternet_hoursday')/pl.col('SDS-SDS_Total_T')).alias('internet_sds_2'), 
                   (pl.col('PreInt_EduHx-computerinternet_hoursday')*pl.col('PAQ_Total')).alias('internet_paq_1'), 
                   (pl.col('PreInt_EduHx-computerinternet_hoursday')/pl.col('PAQ_Total')).alias('internet_paq_2'), 
                ]
            )
        )
        
    def __get_single_time_series_df(self, pl_dataframe: pl.LazyFrame) -> pl.DataFrame:
        result_df: pl.DataFrame = (
            pl_dataframe
            .with_columns(
                (pl.col('X').pow(2) + pl.col('Y').pow(2)).sqrt().alias('2d_norm')
            )
            .group_by(self.config_dict['ID_COL'])
            .agg(
                [
                    (
                        pl.col(col)
                        .filter(pl.col('non-wear_flag')==0)
                        .min()
                        .alias(f'time_series_{col}_min')
                    )
                    for col in self.config_dict['COLUMN_INFO']['TIME_SERIES_FEATURES']
                ] +
                [
                    (
                        pl.col(col)
                        .filter(pl.col('non-wear_flag')==0)
                        .max()
                        .alias(f'time_series_{col}_max')
                    )
                    for col in self.config_dict['COLUMN_INFO']['TIME_SERIES_FEATURES']
                ] +
                [
                    (
                        pl.col(col)
                        .filter(pl.col('non-wear_flag')==0)
                        .mean()
                        .alias(f'time_series_{col}_mean')
                    )
                    for col in self.config_dict['COLUMN_INFO']['TIME_SERIES_FEATURES']
                ] +
                [
                    (
                        pl.col(col)
                        .filter(pl.col('non-wear_flag')==0)
                        .std()
                        .alias(f'time_series_{col}_std')
                    )
                    for col in self.config_dict['COLUMN_INFO']['TIME_SERIES_FEATURES']
                ] +
                [
                    (
                        pl.col(col)
                        .filter(pl.col('non-wear_flag')==0)
                        .median()
                        .alias(f'time_series_{col}_median')
                    )
                    for col in self.config_dict['COLUMN_INFO']['TIME_SERIES_FEATURES']
                ] +
                [
                    (
                        pl.col('relative_date_PCIAT')
                        .filter(pl.col('non-wear_flag')==0)
                        .n_unique()
                        .alias('time_series_total_day_registered')
                    ),
                    (
                        pl.col('non-wear_flag')
                        .mean()
                        .alias('time_series_non-wear_flag_mean')
                    ),
                    (
                        (pl.col('non-wear_flag').count()/(self.total_5s_time_over_day * pl.col('relative_date_PCIAT').n_unique()))
                        .alias('time_series_missing_data_mean')
                    ),
                    pl.col('weekday').mean().alias('time_series_weekday_mean'), 
                    pl.col('quarter').mean().alias('time_series_quarter_mean')
                ]   
            )
            .collect()
        )
        return result_df
    
    def __create_time_series_feature(self) -> None:
        if '2d_norm' not in self.config_dict['COLUMN_INFO']['TIME_SERIES_FEATURES']:
            self.config_dict['COLUMN_INFO']['TIME_SERIES_FEATURES'].append('2d_norm')
        
        self.time_series_list: list[pl.DataFrame] = [
                self.__get_single_time_series_df(pl_dataframe=pl_dataframe)
                for pl_dataframe in tqdm(self.time_series_list, total=len(self.time_series_list)) 
            ]
        
        self.time_series: pl.LazyFrame = pl.concat(self.time_series_list, how='vertical_relaxed').lazy()
        self.preprocess_logger.info(
            f'{len(self._get_col_name(self.time_series))-1} time series columns'
        )
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