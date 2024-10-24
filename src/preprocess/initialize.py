import os
import logging
import polars as pl

from typing import Any, Union

from src.utils.logging_utils import get_logger
from src.base.preprocess.initialize import BaseInit
from src.utils.import_utils import import_json

class PreprocessInit(BaseInit):
    def __init__(self, 
            config_dict: dict[str, Any],
        ):
        self.config_dict: dict[str, Any] = config_dict
        self.n_folds: int = self.config_dict['N_FOLD']

        self.inference: bool = False
        
        self._initialize_all()
        
        if not self.inference:
            self._initialize_preprocess_logger()
        
    def _initialize_all(self) -> None:
        self._initialize_empty_dataset()       
        self._initialize_col_info()
        self._initialize_other_info()
    
    def _initialize_other_info(self) -> None:
        self.id_col: str = self.config_dict['ID_COL']
        self.time_series_list: list[pl.LazyFrame] = []
        
    def _initialize_preprocess_logger(self) -> None:
        self.preprocess_logger: logging.Logger = get_logger('preprocess.txt')
                
    def _initialize_col_info(self) -> None:
        self.target: str = self.config_dict['COLUMN_INFO']['TARGET']
        self.string_mapper: dict[str, dict[str, int]] = import_json(
            os.path.join(
                self.config_dict['PATH_MAPPER_DATA'],
                "mapper_category.json"
            )
        )
        
    def _initialize_empty_dataset(self) -> None:
        self.base_data: Union[pl.LazyFrame, pl.DataFrame]
        self.data: Union[pl.LazyFrame, pl.DataFrame]
        self.time_series: Union[pl.LazyFrame, pl.DataFrame]
        
    def _get_dataframe(self, data: Union[pl.DataFrame, pl.LazyFrame]) -> pl.DataFrame:
        if isinstance(data, pl.LazyFrame):
            return data.collect()
        else:
            return data
        
    def _get_col_name(self, data: Union[pl.DataFrame, pl.LazyFrame]) -> list[str]:
        return data.collect_schema().names()
    
    def _get_number_rows(self, data: Union[pl.DataFrame, pl.LazyFrame]) -> int:
        num_rows = data.select(pl.len())
        
        return self._collect_item_utils(num_rows)
    
    def _collect_item_utils(self, data: Union[pl.DataFrame, pl.LazyFrame]) -> Any:
        if isinstance(data, pl.LazyFrame):
            return data.collect().item()
        else:
            return data.item()