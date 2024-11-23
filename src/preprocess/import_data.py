import os
import polars as pl

from src.base.preprocess.import_data import BaseImport
from src.preprocess.initialize import PreprocessInit

class PreprocessImport(BaseImport, PreprocessInit):
    def scan_all_dataset(self):
        step: str = ("test" if self.inference else "train")
        self.base_data: pl.LazyFrame = (
            pl.scan_csv(
                os.path.join(
                    self.config_dict['PATH_BRONZE_DATA'],
                    self.config_dict['ORIGINAL_DATA_FOLDER'],
                    f'{step}.csv'
                )
            )
        )
        self.__scan_time_series()
        
    def __scan_time_series(self) -> None:
        step: str = ("TEST" if self.inference else "TRAIN")
        folder_to_time_series: str = os.path.join(
            self.config_dict['PATH_BRONZE_DATA'],
            self.config_dict['ORIGINAL_DATA_FOLDER'],
            self.config_dict[f'ORIGINAL_{step}_CHUNK_FOLDER']
        )
        
        list_time_series: list[str] = os.listdir(folder_to_time_series)
        self.time_series_list: list[pl.LazyFrame] = []
        
        for time_series_id_folder in list_time_series:
            id_name: str = time_series_id_folder.split("=")[-1]
            self.time_series_list.append(
                pl.scan_parquet(
                    os.path.join(
                        folder_to_time_series, 
                        time_series_id_folder, 
                        'part-0.parquet'
                    )
                )
                #near the test
                .filter(
                    (pl.col('relative_date_PCIAT').abs()<=30)
                )
                .with_columns(
                    [
                        pl.col('time_of_day').cast(pl.Time),
                        pl.lit(id_name).alias(self.config_dict['ID_COL'])
                    ]
                )
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