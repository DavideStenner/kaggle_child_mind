if __name__=='__main__':
    import os
    import argparse    
    import numpy as np
    import pandas as pd
    import polars as pl
    
    from src.utils.import_utils import import_config, import_params
    from src.preprocess.pipeline import PreprocessPipeline
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ensemble', type=str)
    
    args = parser.parse_args()

    config_dict = import_config()    
    
    test_data = pl.read_parquet(
        os.path.join(
            config_dict['PATH_GOLD_DATA'], 
            'test_data.parquet'
        )
    )
    if (args.model == 'ensemble'):
        from src.model.ensemble.pipeline import EnsemblePipeline
        
        config_dict = import_config()    
        params_model, experiment_name = import_params(model='ensemble')
        
        
        ensemble_pipeline = EnsemblePipeline(
            experiment_name=experiment_name + '_ensemble', params_ensemble=params_model,
            config_dict=config_dict
        )
        ensemble_pipeline.activate_inference()
        
        result_prediction: np.ndarray = ensemble_pipeline.treshold_predict(
            test_data = test_data
        )
        
        submission = (
            test_data.select(config_dict['ID_COL'])
            .to_pandas()
        )
        submission[config_dict['COLUMN_INFO']['TARGET']] = result_prediction
                
        (
            submission
            .to_parquet(
                os.path.join(
                    config_dict['PATH_EXPERIMENT'],
                    experiment_name + "_ensemble",
                    'submission.parquet'
                )
            )
        )
    if (args.model == 'lgb'):
        from src.model.lgbm.pipeline import LgbmPipeline
        
        params_model, experiment_name = import_params(model='lgb')
    
        trainer = LgbmPipeline(
            experiment_name=experiment_name + "_lgb",
            params_lgb=params_model,
            config_dict=config_dict,
            evaluate_shap=False,
        )
        trainer.activate_inference()
        
        result_prediction: np.ndarray = trainer.predict(test_data = test_data, model_type='main')
        
        submission = (
            test_data.select(config_dict['ID_COL'])
            .to_pandas()
        )
        submission[config_dict['COLUMN_INFO']['TARGET']] = result_prediction
                
        (
            submission
            .to_parquet(
                os.path.join(
                    config_dict['PATH_EXPERIMENT'],
                    experiment_name + "_lgb",
                    'submission.parquet'
                )
            )
        )