import gc
import polars as pl

from tqdm import tqdm

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
                ] +
                ##Children's Global Assessment Scale
                [
                    (pl.col('Physical-BMI')/pl.col('CGAS-CGAS_Score')).alias('cgas_bmi'),
                    (pl.col('Physical-Height')/pl.col('CGAS-CGAS_Score')).alias('cgas_height'),
                    (pl.col('Physical-Weight')/pl.col('CGAS-CGAS_Score')).alias('cgas_weight'),
                    (pl.col('PAQ_Total')/pl.col('CGAS-CGAS_Score')).alias('cgas_paq_total'),
                    (pl.col('SDS-SDS_Total_T')/pl.col('CGAS-CGAS_Score')).alias('cgas_sds_total'),
                ] +
                ##Physical Measures
                [
                    (pl.col('Physical-BMI') / pl.col('PreInt_EduHx-computerinternet_hoursday')).alias('phisical_BMI_Internet_Hours'),
                    (pl.col('Physical-Height') / pl.col('PreInt_EduHx-computerinternet_hoursday')).alias('phisical_height_Internet_Hours'),
                    (pl.col('Physical-Weight') / pl.col('PreInt_EduHx-computerinternet_hoursday')).alias('phisical_weight_Internet_Hours'),
                    (pl.col('Physical-Systolic_BP')-pl.col('Physical-Diastolic_BP')).alias('phisical_Pulse_Pressure'),
                    (
                        (pl.col('Physical-Diastolic_BP'))/
                        ((pl.col('Physical-Systolic_BP')-pl.col('Physical-Diastolic_BP'))/3)
                    ).alias('phisical_Mean_Arterial_Pressure')
                ] +                
                ##FitnessGram Vitals and Treadmill
                [
                    (pl.col('Fitness_Endurance-Time_Mins')*60+pl.col('Fitness_Endurance-Time_Sec')).alias('fitnessgram_Fitness_Endurance-Time_Total'),
                    (pl.col('Fitness_Endurance-Max_Stage')/pl.col('PreInt_EduHx-computerinternet_hoursday')).alias('fitnessgram_Fitness_Endurance_internet'),
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
                    (
                        (
                            (pl.col('FGC-FGC_PU')-pl.col('FGC-FGC_PU').mean())/pl.col('FGC-FGC_PU').std() +
                            (pl.col('FGC-FGC_CU')-pl.col('FGC-FGC_CU').mean())/pl.col('FGC-FGC_CU').std() +
                            (pl.col('FGC-FGC_GSD')-pl.col('FGC-FGC_GSD').mean())/pl.col('FGC-FGC_GSD').std() +
                            (pl.col('FGC-FGC_GSND')-pl.col('FGC-FGC_GSND').mean())/pl.col('FGC-FGC_GSND').std()
                        )/4
                    ).alias('fitnessgramchildOverall_Strength_Score'),
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
                    ).alias('fitnessgramchild_Overall_FGC_score'),
                    (pl.col('FGC-FGC_SRR') + pl.col('FGC-FGC_SRL') + pl.col('FGC-FGC_TL')).alias('fitnessgramchild_Overall_Flexibility_Score'),
                ] +
                
                ##Bio-electric Impedance Analysis
                [
                    (pl.col('BIA-BIA_Fat') / pl.col('BIA-BIA_BMI')).alias('bio_BFP_BMI'),
                    (pl.col('BIA-BIA_FFMI') / pl.col('BIA-BIA_Fat')).alias('bio_FFMI_BFP'),
                    (pl.col('BIA-BIA_FMI') / pl.col('BIA-BIA_Fat')).alias('bio_FMI_BFP'),
                    (pl.col('BIA-BIA_LST') / pl.col('BIA-BIA_TBW')).alias('bio_LST_TBW'),
                    (pl.col('BIA-BIA_Fat') * pl.col('BIA-BIA_BMR')).alias('bio_BFP_BMR'),
                    (pl.col('BIA-BIA_Fat') * pl.col('BIA-BIA_DEE')).alias('bio_BFP_DEE'),
                    (pl.col('BIA-BIA_SMM') / pl.col('BIA-BIA_FMI')).alias('bio_Muscle_to_Fat'),
                    (pl.col('BIA-BIA_ICW') / pl.col('BIA-BIA_TBW')).alias('bio_ICW_TBW'),
                    
                    (pl.col('BIA-BIA_ECW') / pl.col('BIA-BIA_ICW')).alias('bio_Water_Balance_Ratio'),
                    (pl.col('BIA-BIA_BMR') / pl.col('Physical-Weight')).alias('bio_BMR_Weight'),
                    
                    (pl.col('BIA-BIA_DEE') / pl.col('Physical-Weight')).alias('bio_DEE_Weight'),
                    (pl.col('BIA-BIA_SMM') / pl.col('Physical-Height')).alias('bio_SMM_Height'),
                    (pl.col('BIA-BIA_TBW') / pl.col('Physical-Weight')).alias('bio_Hydration_Status'),
                ] +
                ## Physical Activity Questionnaire - Sleep Disturbance Scale
                [
                    (
                       pl.col('PAQ_Total')/pl.col('SDS-SDS_Total_T')
                    ).alias('sleep_activit_sleep_interaction')        
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
                            (
                                pl.col(col)
                                .filter(pl.col('non-wear_flag')==0)
                                .count()
                                .alias(f'time_series_{col}_count')
                            )
                            for col in self.config_dict['COLUMN_INFO']['TIME_SERIES_FEATURES']
                        ] +
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
                                .median()
                                .alias(f'time_series_{col}_median')
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