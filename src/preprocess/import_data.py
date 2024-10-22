import os
import polars as pl

from src.base.preprocess.import_data import BaseImport
from src.preprocess.initialize import PreprocessInit

class PreprocessImport(BaseImport, PreprocessInit):
    def scan_all_dataset(self):
        self.base_data: pl.LazyFrame = (
            pl.scan_csv(
                os.path.join(
                    self.config_dict['PATH_BRONZE_DATA'],
                    self.config_dict['ORIGINAL_DATA_FOLDER'],
                    'train.csv'
                )
            )
            .filter(pl.col(self.target).is_not_null())
        )
                

    def downcast_data(self):
        self.base_data = (
            self.base_data
            .with_columns(
                [
                    pl.col(col).cast(pl.UInt8)
                    for col in self.config_dict['COLUMN_INFO']['CATEGORICAL_FEATURE']
                ] +
                [
                    pl.col(col).replace(self.string_mapper['main'][col]).cast(pl.UInt8)
                    for col in self.config_dict['COLUMN_INFO']['STRING_FEATURE']
                ]
            )
        )

        if not self.inference:
            self.base_data = (
                self.base_data
                .with_columns(pl.col(self.target).cast(pl.UInt8))
            )
            
    def import_all(self) -> None:
        self.scan_all_dataset()
        self.downcast_data()