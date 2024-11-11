if __name__=='__main__':
    import argparse
    import warnings
    import matplotlib.pyplot as plt

    from src.utils.import_utils import import_config, import_params, import_json

    #filter useless warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    
    plt.set_loglevel('WARNING')
    
    config_dict = import_config()    
    params_model, experiment_name = import_params(model='ensemble')
    from src.model.ensemble.pipeline import EnsemblePipeline
    
    ensemble_pipeline = EnsemblePipeline(
        experiment_name=experiment_name + '_ensemble', params_ensemble=params_model,
        config_dict=config_dict
    )
    ensemble_pipeline.ensemble_preprocess()
