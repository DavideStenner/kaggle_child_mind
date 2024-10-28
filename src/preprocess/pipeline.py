import os
import gc

import polars as pl

from typing import Any, Tuple

from src.base.preprocess.pipeline import BasePipeline
from src.preprocess.import_data import PreprocessImport
from src.preprocess.initialize import PreprocessInit
from src.preprocess.add_feature import PreprocessAddFeature
from src.preprocess.cv_fold import PreprocessFoldCreator

class PreprocessPipeline(BasePipeline, PreprocessImport, PreprocessAddFeature, PreprocessFoldCreator):

    def __init__(self, config_dict: dict[str, Any]):
                
        PreprocessInit.__init__(
            self, 
            config_dict=config_dict, 
        )

    def save_data(self) -> None:       
        self.preprocess_logger.info('saving processed dataset')
        
        (
            self.data
            .filter(pl.col(self.target).is_not_null())
            .write_parquet(
            os.path.join(
                    self.config_dict['PATH_GOLD_DATA'],
                    f'data.parquet'
                )
            )
        )
        
        (
            self.null_data
            .write_parquet(
            os.path.join(
                    self.config_dict['PATH_GOLD_DATA'],
                    f'data_null.parquet'
                )
            )
        )
            
    def collect_feature(self) -> None:
        self.base_data: pl.DataFrame = self._get_dataframe(self.base_data)
        
    def collect_all(self) -> None:
        self.collect_feature()
        
        
    def preprocess_inference(self) -> None:
        self.preprocess_logger.info('Creating feature')
        self.create_feature()

        self.preprocess_logger.info('Merging all')
        self.merge_all()

        self.preprocess_logger.info('Collecting Dataset')

        self.data: pl.DataFrame = self._get_dataframe(self.data)
        self.preprocess_logger.info(
            f'Collected dataset with {len(self._get_col_name(self.data))} columns and {self._get_number_rows(self.data)} rows'
        )

        self.preprocess_logger.info('Saving test dataset')
        self.data.write_parquet(
            os.path.join(
                self.config_dict['PATH_GOLD_PARQUET_DATA'],
                f'test_data.parquet'
            )
        )
        _ = gc.collect()

    def preprocess_train(self) -> None:        
        self.preprocess_logger.info('beginning preprocessing training dataset')
        self.preprocess_logger.info('Creating feature')
        self.create_feature()

        self.preprocess_logger.info('Merging all')
        self.merge_all()
        
        self.preprocess_logger.info('Collecting Dataset')
        
        self.data: pl.DataFrame = self._get_dataframe(self.data)
        self.preprocess_logger.info(
            f'Collected dataset with {len(self._get_col_name(self.data))} columns and {self._get_number_rows(self.data)} rows'
        )

        _ = gc.collect()
        
        self.preprocess_logger.info('Creating fold_info column ...')
        self.create_fold()
        
        self.save_data()
                
    def begin_inference(self) -> None:
        self.preprocess_logger.info('beginning preprocessing inference dataset')
        
        #reset data
        self.base_data = None
        self.data = None
        
        self.inference: bool = True

    def __call__(self) -> None:
        self.import_all()
        
        if self.inference:    
            raise NotImplementedError

        else:
            self.preprocess_train()
