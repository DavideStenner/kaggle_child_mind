call .venv\Scripts\activate
python preprocess.py
python train.py --model lgb
python train.py --model xgb
python train.py --model ctb
python ensemble.py