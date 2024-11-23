import torch
import sklearn

import numpy as np
import numpy as np
import pandas as pd
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer




class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.ordinal_enc_ = OrdinalEncoder(unknown_value=np.NaN, encoded_missing_value=np.NaN,
                                           handle_unknown='use_encoded_value')
        self.ordinal_enc_.fit(X)
        self.cat_sizes_ = []
        for cat_arr in self.ordinal_enc_.categories_:
            has_nan = np.any([isinstance(val, (float, np.float32, np.float64)) and np.isnan(val) for val in cat_arr])
            self.cat_sizes_.append(len(cat_arr) - int(has_nan))

        return self

    def transform(self, X, y=None):
        x_enc = self.ordinal_enc_.transform(X)
        n_samples = x_enc.shape[0]
        out_arrs = []
        for i, cat_size in enumerate(self.cat_sizes_):
            column = x_enc[:, i]
            idxs = np.arange(n_samples)
            isnan = np.isnan(column)
            out_arr = np.zeros(shape=(n_samples, cat_size))
            # do one-hot encoding, encode nan (missing or unknown) values to all zeros
            out_arr[idxs[~isnan], column[~isnan].astype(np.int64)] = 1.

            if cat_size == 2:
                # binary: encode to single feature being -1, 1 or 0 (for missing or unknown values)
                out_arr = out_arr[:, 0:1] - out_arr[:, 1:2]

            out_arrs.append(out_arr)

        return np.concatenate(out_arrs, axis=-1)


class CustomOneHotPipeline(BaseEstimator, TransformerMixin):
    """
    Apply CustomOneHotEncoder only to categorical features.
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.tfm_ = ColumnTransformer(transformers=[
            ('categorical', CustomOneHotEncoder(),
             make_column_selector(dtype_include=["string", "object", "category"]))
            # ('remaining', FunctionTransformer(lambda x: x),
            #  make_column_selector(dtype_exclude=["string", "object", "category"]))
        ]).fit(X)
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X)
        return self.tfm_.transform(X)


class RobustScaleSmoothClipTransform(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def fit(self, X, y=None):
        # don't deal with dataframes for simplicity
        assert isinstance(X, np.ndarray)
        self._median = np.median(X, axis=-2)
        quant_diff = np.quantile(X, 0.75, axis=-2) - np.quantile(X, 0.25, axis=-2)
        max = np.max(X, axis=-2)
        min = np.min(X, axis=-2)
        idxs = quant_diff == 0.0
        # on indexes where the quantile difference is zero, do min-max scaling instead
        quant_diff[idxs] = 0.5 * (max[idxs] - min[idxs])
        factors = 1.0 / (quant_diff + 1e-30)
        # if feature is constant on the training data,
        # set factor to zero so that it is also constant at prediction time
        factors[quant_diff == 0.0] = 0.0
        self._factors = factors
        return self

    def transform(self, X, y=None):
        x_scaled = self._factors[None, :] * (X - self._median[None, :])
        x_scaled = x_scaled / np.sqrt(1 + (x_scaled / 3) ** 2)
        return np.nan_to_num(x_scaled)

def get_realmlp_td_s_pipeline():
    return sklearn.pipeline.Pipeline([('one_hot', CustomOneHotPipeline()),
                                      ('rssc', RobustScaleSmoothClipTransform())])

class ScalingLayer(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale[None, :]


class NTPLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, zero_init: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factor = 0.0 if zero_init else 1.0
        self.weight = nn.Parameter(factor * torch.randn(in_features, out_features))
        self.bias = nn.Parameter(factor * torch.randn(1, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (1. / np.sqrt(self.in_features)) * (x @ self.weight) + self.bias


class Mish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul(torch.tanh(torch.nn.functional.softplus(x)))

class SimpleMLP(BaseEstimator):
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.log_print: int = 10
        
    def __l2_metric(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return (y_pred - y_true).square().mean()
        
    def get_model(self, input_dim: int) -> nn.Sequential:
        act = Mish
        model = nn.Sequential(
            ScalingLayer(input_dim),
            NTPLinear(input_dim, input_dim*2), act(),
            NTPLinear(input_dim*2, input_dim*4), act(),
            NTPLinear(input_dim*4, input_dim*2), act(),
            NTPLinear(input_dim*2, 1, zero_init=True),
        ).to(self.device)
        return model
    
    def fit(self, train_set: Tuple[np.ndarray], valid_set: Tuple[np.ndarray]) -> Tuple[dict[str, list], list]:
        train_x, train_y = train_set
        valid_x, valid_y = valid_set

        input_dim = train_x.shape[1]

        self.y_mean_ = np.mean(train_y, axis=0)
        self.y_std_ = np.std(train_y, axis=0)
        
        train_y = (train_y - self.y_mean_) / (self.y_std_ + 1e-30)        
        valid_y = (valid_y - self.y_mean_) / (self.y_std_ + 1e-30)
        
        model = self.get_model(input_dim=input_dim)
        criterion = nn.MSELoss()
        params = list(model.parameters())
        
        scale_params = [params[0]]
        weights = params[1::2]
        biases = params[2::2]
        
        opt = torch.optim.Adam([dict(params=scale_params), dict(params=weights), dict(params=biases)],
                                betas=(0.9, 0.95))

        x_train = torch.as_tensor(train_x, dtype=torch.float32)
        y_train = torch.as_tensor(train_y, dtype=torch.float32)

        x_valid = torch.as_tensor(valid_x, dtype=torch.float32)
        y_valid = torch.as_tensor(valid_y, dtype=torch.float32)

        if len(y_train.shape) == 1:
            y_train = y_train[:, None]

        if len(y_valid.shape) == 1:
            y_valid = y_valid[:, None]

        train_ds = TensorDataset(x_train, y_train)
        valid_ds = TensorDataset(x_valid, y_valid)
        
        n_train = x_train.shape[0]
        n_valid = x_valid.shape[0]
        
        n_epochs = 256
        
        train_batch_size = min(256, n_train)
        valid_batch_size = max(1, min(1024, n_valid))

        train_dl = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, drop_last=True)
        valid_dl = DataLoader(valid_ds, batch_size=valid_batch_size, shuffle=False)

        n_train_batches = len(train_dl)
        base_lr = 0.0001
        
        history_loss = {
            'l2': [],
        }
        history_param = []

        for epoch in range(n_epochs):

            for batch_idx, (x_batch, y_batch) in enumerate(train_dl):
                # set learning rates according to schedule
                t = (epoch * n_train_batches + batch_idx) / (n_epochs * n_train_batches)
                lr_sched_value = 0.5 - 0.5 * np.cos(2 * np.pi * np.log2(1 + 15 * t))
                lr = base_lr * lr_sched_value

                opt.param_groups[0]['lr'] = 6 * lr  # for scale
                opt.param_groups[1]['lr'] = lr  # for weights
                opt.param_groups[2]['lr'] = 0.1 * lr  # for biases

                # optimization
                y_pred = model(x_batch.to(self.device))
                loss = criterion(y_pred, y_batch.to(self.device))
                loss.backward()
                opt.step()
                opt.zero_grad()

            # save parameters if validation score improves
            with torch.no_grad():
                y_pred_valid = torch.cat(
                    [
                        model(valid_x_batch.to(self.device)).detach() 
                        for valid_x_batch, _ in valid_dl
                    ], 
                    dim=0
                )

                valid_l2_loss = self.__l2_metric(y_pred = y_pred_valid, y_true=y_valid).cpu().item()
                
                history_loss['l2'].append(valid_l2_loss)

                history_param.append(
                    [p.detach().clone() for p in model.parameters()]
                )
            if epoch > 0:
                if epoch % self.log_print == 0:
                    print(f'[{epoch}]  valid-l2: {valid_l2_loss:.5f}')
        
        return history_loss, history_param
    
    def load_best_param(self, best_valid_params: list, input_dim: int) -> None:
        model = self.get_model(input_dim=input_dim)
        with torch.no_grad():
            for p_model, p_copy in zip(model.parameters(), best_valid_params):
                p_model.set_(p_copy)
        
        self.model_ = model
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        
        x = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        self.model_.eval()
        with torch.no_grad():
            y_pred = self.model_(x).cpu().numpy()

        return y_pred[:, 0] * self.y_std_ + self.y_mean_

class Standalone_RealMLP_TD_S_Regressor(BaseEstimator, RegressorMixin):
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.prep_ = get_realmlp_td_s_pipeline()
        
    def fit(self, train_set: Tuple[pd.DataFrame], valid_set: Tuple[pd.DataFrame]) -> Tuple[dict[str, list], list]:
        model_ = SimpleMLP(device=self.device)
        train_x, train_y = train_set
        valid_x, valid_y = valid_set

        train_x = self.prep_.fit_transform(train_x)
        valid_x = self.prep_.transform(valid_x)

        history_loss, history_param = model_.fit(
            (train_x, train_y.to_numpy()), 
            (valid_x, valid_y.to_numpy())
        )
        self.model_ = model_
        return history_loss, history_param
    
    def load_model(self, best_valid_params: list, input_dim=int) -> None:
        self.model_.load_best_param(best_valid_params=best_valid_params, input_dim=input_dim)
        
    def predict(self, X) -> np.ndarray:
        return self.model_.predict(self.prep_.transform(X))