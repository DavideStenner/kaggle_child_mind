import os
import shap

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt

from tqdm import tqdm
from functools import partial
from typing import Union, Dict, Tuple
from src.model.lgbm.initialize import LgbmInit
from src.model.metric.official_metric import quadratic_weighted_kappa, quadratic_weighted_kappa_tresh

class LgbmExplainer(LgbmInit):       
    def plot_train_curve(self, 
            progress_df: pd.DataFrame, 
            variable_to_plot: Union[str, list],  metric_to_eval: str,
            name_plot: str, model_type: str,
            best_epoch_lgb:int
        ) -> None:
        
        if isinstance(variable_to_plot, str):
            variable_to_plot = [variable_to_plot]
                        
        fig = plt.figure(figsize=(18,8))
        sns.lineplot(
            data=progress_df[['time'] + variable_to_plot].melt(
                id_vars='time',
                value_vars=variable_to_plot,
                var_name='metric_fold', value_name=metric_to_eval
            ), 
            x="time", y=metric_to_eval, hue='metric_fold'
        )
        plt.axvline(x=best_epoch_lgb, color='blue', linestyle='--')

        plt.title(f"Training plot curve of {metric_to_eval}")

        fig.savefig(
            os.path.join(
                self.experiment_path_dict['training'].format(model_type=model_type),
                f'{name_plot}.png'
            )
        )
        plt.close(fig)

    def evaluate_score(self) -> None:    
        for model_type in self.model_used:
            self.__evaluate_single_model(model_type=model_type)
        
    def __evaluate_single_model(self, model_type: str) -> None:
        metric_eval = self.model_metric_used[model_type]['label']
        metric_to_max = self.model_metric_used[model_type]['maximize']
        
        #load feature list
        self.load_used_feature(model_type=model_type)
        
        # Find best epoch
        progress_list = self.load_progress_list(
            model_type=model_type
        )

        progress_dict = {}

        list_metric = progress_list[0]['valid'].keys()
        
        for metric_ in list_metric:
            progress_dict.update(
                {
                    f"{metric_}_fold_{i}": progress_list[i]['valid'][metric_]
                    for i in range(self.n_fold)
                }
            )

        progress_df = pd.DataFrame(progress_dict)
        progress_df['time'] = range(progress_df.shape[0])
        
        for metric_ in list_metric:
            
            progress_df[f"average_{metric_}"] = progress_df.loc[
                :, [metric_ in x for x in progress_df.columns]
            ].mean(axis =1)
        
            progress_df[f"std_{metric_}"] = progress_df.loc[
                :, [metric_ in x for x in progress_df.columns]
            ].std(axis =1)

        if metric_to_max:
            best_epoch_lgb = int(progress_df[f"average_{metric_eval}"].argmax())
        else:
            best_epoch_lgb = int(progress_df[f"average_{metric_eval}"].argmin())

        best_score_lgb = progress_df.loc[
            best_epoch_lgb,
            f"average_{metric_eval}"
        ]
        lgb_std = progress_df.loc[
            best_epoch_lgb, f"std_{metric_eval}"
        ]

        self.training_logger.info(f'{model_type} Best epoch: {best_epoch_lgb}, CV-{metric_eval}: {best_score_lgb:.5f} ± {lgb_std:.5f}')

        best_result = {
            'best_epoch': best_epoch_lgb+1,
            'best_score': best_score_lgb,
            'all_best_score': {
                metric_: progress_df.loc[
                    best_epoch_lgb,
                    f"average_{metric_}"
                ]
                for metric_ in list_metric
            }
        }
        
        for metric_ in list_metric:
            #plot cv score
            self.plot_train_curve(
                progress_df=progress_df, 
                variable_to_plot=f'average_{metric_}', metric_to_eval=metric_,
                name_plot=f'average_{metric_}_training_curve', model_type=model_type,
                best_epoch_lgb=best_epoch_lgb
            )
            #plot every fold score
            self.plot_train_curve(
                progress_df=progress_df, 
                variable_to_plot=[f'{metric_}_fold_{x}' for x in range(self.n_fold)],
                metric_to_eval=metric_,
                name_plot=f'training_{metric_}_curve_by_fold', model_type=model_type,
                best_epoch_lgb=best_epoch_lgb
            )

        #plot std score
        self.plot_train_curve(
            progress_df=progress_df, 
            variable_to_plot=f'std_{metric_eval}', metric_to_eval=metric_eval,
            name_plot='std_training_curve', model_type=model_type,
            best_epoch_lgb=best_epoch_lgb
        )
        
        self.save_best_result(
            best_result=best_result, model_type=model_type, 
        )
            
    def get_feature_importance(self) -> None:        
        for model_type in self.model_used:
            self.load_used_feature(model_type=model_type)
            self.__get_single_feature_importance(model_type=model_type)
    
    def __get_single_feature_importance(self, model_type: str) -> None:
        best_result = self.load_best_result(
            model_type=model_type
        )
        model_list: list[lgb.Booster] = self.load_pickle_model_list(
            model_type=model_type, 
        )

        feature_importances = pd.DataFrame()
        feature_importances['feature'] = self.feature_list

        for fold_, model in enumerate(model_list):
            feature_importances[f'fold_{fold_}'] = model.feature_importance(
                importance_type='gain', iteration=best_result['best_epoch']
            )

        feature_importances['average'] = feature_importances[
            [f'fold_{fold_}' for fold_ in range(self.n_fold)]
        ].mean(axis=1)
        
        feature_importances = (
            feature_importances[['feature', 'average']]
            .sort_values(by='average', ascending=False)
        )
        self.training_logger.info(
            f"Model {model_type} top 5 features are {', '.join(feature_importances['feature'].iloc[:5])}"
        )
        #plain feature
        fig = plt.figure(figsize=(18,8))
        sns.barplot(data=feature_importances.head(50), x='average', y='feature')
        plt.title(f"{model_type} 50 TOP feature importance over {self.n_fold} average")

        fig.savefig(
            os.path.join(
                self.experiment_path_dict['feature_importance'].format(model_type=model_type), 
                'importance_plot.png'
            )
        )
        plt.close(fig)
        
        #feature importance excel
        feature_importances.to_excel(
            os.path.join(
                self.experiment_path_dict['feature_importance'].format(model_type=model_type), 
                'feature_importances.xlsx'
            ),
            index=False
        )

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
                .select(self.feature_list)
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
        self.training_logger.info(f'{len(self.list_treshold_value)} postprocess treshold combinations')
        dataset_list: list[Tuple[pd.DataFrame, pd.DataFrame]] = self.__get_list_of_oof_dataset()

        for model_type in self.model_used:
            model_list: list[lgb.Booster] = self.load_pickle_model_list(
                model_type=model_type, 
            )
            best_result: dict[str, any] = self.load_best_result(model_type=model_type)
            
            best_epoch: int = best_result['best_epoch']

            result_by_combination: np.ndarray = np.zeros(len(self.list_treshold_value))
            
            self.training_logger.info(f'Starting {model_type} treshold postprocess')

            for fold_, (test_feature, test_target) in enumerate(dataset_list):
                pred_target = model_list[fold_].predict(
                    data=test_feature.to_numpy('float64'),
                    num_iteration = best_epoch
                )
                treshold_iterator = tqdm(self.list_treshold_value, total=len(self.list_treshold_value))
                
                #partial for efficiency
                partial_kappa = partial(
                    quadratic_weighted_kappa_tresh,
                    y_true=test_target,
                    y_pred=pred_target
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
            
            base_cv_score = best_result['all_best_score']['q_kappa']
            increment_score = best_score - base_cv_score
            
            self.training_logger.info(
                f'Best treshold combination for {model_type} with optimized score of {best_score:.6f} with an increment of {increment_score:.6f}:'
            )
            self.training_logger.info(
                f'T1: {best_combination[0]}\nT2: {best_combination[1]}\nT3: {best_combination[2]}'
            )
            best_result['treshold_optim'] = {
                'best_score': best_score,
                'best_combination': best_combination
            }
            
            self.save_best_result(
                best_result=best_result, model_type=model_type, 
            )

    def oof_get_shap_contribution(self) -> None:
        np.seterr(invalid='ignore')
        
        self.training_logger.info('Starting shap calculation')        
        dataset_list: list[Tuple[pd.DataFrame, np.ndarray]] = self.__get_list_of_oof_dataset()

        for model_type in self.model_used:
            self.load_used_feature(model_type=model_type)

            model_list: list[lgb.Booster] = self.load_pickle_model_list(
                model_type=model_type, 
            )
            best_result: dict[str, any] = self.load_best_result(model_type=model_type)
            best_epoch: int = best_result['best_epoch']
            
            shap_list: list[np.ndarray] = []
            data_list: list[np.ndarray] = []
            shap_insight: pd.DataFrame = pd.DataFrame(
                {
                    'feature': self.feature_list
                }
            )            
            for fold_, (test_feature, _) in enumerate(dataset_list):          
                shap_values = model_list[fold_].predict(
                    test_feature, 
                    num_iteration=best_epoch, pred_contrib=True
                )[:, :-1]
                
                data_list.append(test_feature)
                shap_list.append(shap_values)
                shap_insight[f'imp_shap_fold_{fold_}'] = np.abs(shap_values).mean(axis=0)
            
            shap_values = np.concatenate(shap_list, axis=0)
            data_values = np.concatenate(data_list, axis=0)
            shap_insight[f'mean_imp_shap'] = shap_insight[
                [f'imp_shap_fold_{fold_}' for fold_ in range(self.n_fold)]
            ].mean(axis=1)
        
            (
                shap_insight
                .sort_values('mean_imp_shap', ascending=False)
                .reset_index(drop=True)
                .to_excel(
                    os.path.join(
                        self.experiment_path_dict['feature_importance'].format(model_type=model_type), 
                        'shap_importance.xlsx'
                    ),
                    index=False
                )
            )
                
            #shap
            fig = plt.figure()
            shap.summary_plot(
                shap_values, data_values,
                [
                    col[:30] for col in self.feature_list
                ],
                show=False, 
                max_display=30
            )
            plt.savefig(
                os.path.join(
                    self.experiment_path_dict['insight'].format(model_type=model_type),
                    f'shap_insight.png'
                )
            )
            plt.close(fig)

            #begin for data with time series
            idx_row_with_time_series: np.ndarray = (
                np.logical_not(
                    np.isnan(
                        data_values[
                            :,
                            ['time_series_' == col[:12] for col in self.feature_list]
                        ]
                    )
                ).sum(axis=1)>0
            )

            if any(idx_row_with_time_series):
                #shap
                fig = plt.figure()
                shap.summary_plot(
                    shap_values[idx_row_with_time_series, :], data_values[idx_row_with_time_series, :],
                    [
                        col[:30] for col in self.feature_list
                    ],
                    show=False, 
                    max_display=30
                )
                plt.savefig(
                    os.path.join(
                        self.experiment_path_dict['insight'].format(model_type=model_type),
                        f'shap_insight_ts.png'
                    )
                )
                plt.close(fig)
            
        np.seterr(invalid='warn')

    def get_oof_insight(self) -> None:                
        self.oof_get_best_treshold()
        
    def get_oof_prediction(self) -> None:
        self.oof_get_shap_contribution()