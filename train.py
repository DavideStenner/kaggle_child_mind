if __name__=='__main__':
    import argparse
    import warnings
    import matplotlib.pyplot as plt

    from src.utils.import_utils import import_config, import_params, import_json

    #filter useless warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    
    plt.set_loglevel('WARNING')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='lgb', type=str)
    parser.add_argument('--all_model', action='store_true')
    
    args = parser.parse_args()

    config_dict = import_config()    
    
    if (args.model == 'lgb') | (args.all_model):
        from src.model.lgbm.pipeline import LgbmPipeline
        
        params_model, experiment_name = import_params(model='lgb')
        pipeline_params = import_json('config/params_pipeline.json')
        
        updated_config = config_dict.copy()
        updated_config.update(pipeline_params)
        
        trainer = LgbmPipeline(
            experiment_name=experiment_name + "_lgb",
            params_lgb=params_model,
            config_dict=updated_config,
            evaluate_shap=True
        )
        trainer.train_explain()
        
    if (args.model == 'xgb') | (args.all_model):
        from src.model.xgbm.pipeline import XgbPipeline

        params_model, experiment_name = import_params(model='xgb')
        pipeline_params = import_json('config/params_pipeline.json')
        
        updated_config = config_dict.copy()
        updated_config.update(pipeline_params)

        trainer = XgbPipeline(
            experiment_name=experiment_name + "_xgb",
            params_xgb=params_model,
            config_dict=updated_config,
            evaluate_shap=True,
        )
        trainer.train_explain()
        
    if (args.model == 'ctb') | (args.all_model):
        from src.model.ctb.pipeline import CtbPipeline

        params_model, experiment_name = import_params(model='ctb')
        pipeline_params = import_json('config/params_pipeline.json')
        
        updated_config = config_dict.copy()
        updated_config.update(pipeline_params)

        trainer = CtbPipeline(
            experiment_name=experiment_name + "_ctb",
            params_ctb=params_model,
            config_dict=updated_config,
            evaluate_shap=True,
        )
        trainer.train_explain()
    
    # if (args.model == 'nn') | (args.all_model):
    #     from src.model.nn.pipeline import MLPPipeline

    #     params_model, experiment_name = import_params(model='nn')
    #     pipeline_params = import_json('config/params_pipeline.json')
        
    #     updated_config = config_dict.copy()
    #     updated_config.update(pipeline_params)

    #     trainer = MLPPipeline(
    #         experiment_name=experiment_name + "_nn",
    #         params_nn=params_model,
    #         config_dict=updated_config,
    #     )
    #     trainer.train_explain()