import numpy as np

def get_ordinal_target(target_array: np.ndarray, num_target: int) -> np.ndarray:
    ordinal_train_target = np.zeros((target_array.shape[0], num_target))
    for i in range(target_array.shape[0]):
        ordinal_train_target[i, :(target_array[i]+1)] = 1
        
    return ordinal_train_target