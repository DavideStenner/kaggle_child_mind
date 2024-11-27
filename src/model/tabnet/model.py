import pickle
from pytorch_tabnet.callbacks import Callback

class TabNetDumpModel(Callback):
    def __init__(self, ):
        super().__init__()  # Initialize parent class
        
    def on_train_begin(self, logs=None):
        self.model = self.trainer  # Use trainer itself as model
        self.model_list: list[bytes] = []
        
    def on_epoch_end(self, epoch, logs=None):
        self.model_list.append(
            pickle.dumps(self.model.network.state_dict())
        )
