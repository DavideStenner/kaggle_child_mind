import os
import re
import copy
import numpy as np
import polars as pl
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Union, Tuple, Dict
from sklearn.metrics import f1_score
from src.model.xgbm.initialize import XgbInit
from src.model.metric.utils import get_ordinal_target
from src.model.metric.official_metric import xgb_quadratic_kappa, xgb_ordinal_kappa

class XgbExplainer(XgbInit):       
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
            if (
                'multi_strategy' in self.params_xgb.keys()
            ):
                #not supported
                self.__get_permutation_importance(model_type=model_type)
        
            else:
                self.__get_single_feature_importance(model_type=model_type)
    
    def __get_single_feature_importance(self, model_type: str) -> None:
        best_result = self.load_best_result(
            model_type=model_type
        )
        model_list: list[xgb.Booster] = self.load_pickle_model_list(
            model_type=model_type, 
        )

        feature_importances = pd.DataFrame()
        feature_importances['feature'] = self.feature_list

        for fold_, model in enumerate(model_list):
            importance_dict = model.get_score(
                importance_type='gain'
            )

            feature_importances[f'fold_{fold_}'] = (
                feature_importances['feature'].map(importance_dict)
            ).fillna(0)

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
    
    def __calculate_score(self, y_pred:np.ndarray, y_true=np.ndarray) -> float:
        return xgb_ordinal_kappa(y_true=y_true, y_pred=y_pred)
    
    def __oof_score(
            self, 
            dataset_list: list[Tuple[pd.DataFrame, pd.DataFrame]],
            model_list: list[xgb.Booster], best_epoch: int
        ) -> float:
        """
        Get oof score on each validation set, using trained model with correct number of epoch.
        Score over each target and mean of all target also

        Args:
            dataset_list (list[Tuple[pd.DataFrame, pd.DataFrame]]): list of oof dataset feature/target

        Returns:
            float: cv oof score
        """
        score_oof = 0

        for fold_, (test_feature, test_target) in enumerate(dataset_list):
            pred_target = model_list[fold_].predict(
                data=xgb.DMatrix(
                    data=test_feature.to_numpy('float64'),
                    feature_names=self.feature_list
                ),
                iteration_range=(0, best_epoch)
            )
            score_fold = self.__calculate_score(y_pred=pred_target, y_true=test_target)

            score_oof += score_fold/self.n_fold

        return score_oof

    def __get_list_of_oof_dataset(self) -> list[Tuple[pd.DataFrame, pd.DataFrame]]:
        list_dataset: list[Tuple[pd.DataFrame, pd.DataFrame]] = []
        
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
            
            test_target = get_ordinal_target(
                target_array=(
                    fold_data
                    .select(self.target_col)
                    .to_pandas()
                    .to_numpy('int').reshape((-1))
                ),
                num_target=self.config_dict['COLUMN_INFO']['TARGET_N_UNIQUE']
            )
            list_dataset.append([test_feature, test_target])
        
        return list_dataset
    
    def __get_permutation_importance(self, model_type: str) -> None:
        self.__get_multi_permutation_feature_importance(model_type=model_type)
        
    def __shuffled_dataset(self, dataset: pd.DataFrame, feature: str) -> pd.DataFrame:
        dataset[feature] = dataset[feature].sample(frac=1).to_numpy('float64')
        return dataset
    
    def __get_multi_permutation_feature_importance(
            self, 
            model_type:str,
            #magical number
            num_repetition: int = 3
        ) -> None:
        """
        Permutation feature importance in cross validation

        Args:
            num_repetition (int, optional): how many times to repeat each fold shuffle. Defaults to 3.
        """
        model_list: list[xgb.Booster] = self.load_pickle_model_list(
            model_type=model_type
        )
        dataset_list: list[Tuple[pd.DataFrame, pd.DataFrame]] = self.__get_list_of_oof_dataset()
        best_epoch: int = self.load_best_result(model_type=model_type)['best_epoch']
        
        base_score: float = self.__oof_score(
            dataset_list=dataset_list,
            model_list=model_list, best_epoch=best_epoch
        )
        self.training_logger.info(f'{model_type} has a base score of {base_score}')
                
        feature_importance_dict = {
            feature: base_score
            for feature in self.feature_list
        }
        self.training_logger.info(f'Starting {model_type} to calculate permutation importance over {len(self.feature_list)} features')
        for feature in tqdm(self.feature_list):
            shuffled_dataset = copy.deepcopy(dataset_list)

            for _ in range(num_repetition):                
                shuffled_dataset = [
                    [
                        self.__shuffled_dataset(
                            feature_dataset, feature
                        ), 
                        target_dataset
                    ]
                    for feature_dataset, target_dataset in shuffled_dataset
                ]
                result_shuffling = self.__oof_score(
                    dataset_list=shuffled_dataset,
                    model_list=model_list, best_epoch=best_epoch
                )
                feature_importance_dict[feature] -= (
                    result_shuffling/num_repetition
                )

        feature_importance_dict = [
            {
                'feature': feature,
                'importance': change_score
            }
            for feature, change_score in feature_importance_dict.items()
        ]
        result = pd.DataFrame(
            data=feature_importance_dict
        )
        result.sort_values('importance', ascending=False).to_excel(
            os.path.join(
                self.experiment_path_dict['feature_importance'].format(model_type=model_type),
                'feature_importances.xlsx'
            ), 
            index=False
        )
    
    def get_oof_insight(self) -> None:
        pass

    def get_oof_prediction(self) -> None:
        pass