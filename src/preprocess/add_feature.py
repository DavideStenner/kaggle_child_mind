import pandas as pd
import polars as pl

from typing import Dict, Tuple
from itertools import product
from src.base.preprocess.add_feature import BaseFeature
from src.preprocess.initialize import PreprocessInit

class PreprocessAddFeature(BaseFeature, PreprocessInit):

    def create_feature(self) -> None:   
        pass

    def merge_all(self) -> None:
        self.data = self.base_data