# Overview
Competition to predict problematic internet usage in children using physical activity data. Build a model to analyze fitness metrics as early warning signs, enabling timely interventions for healthier digital habits.

https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use

# Set up data

Create the required environment by executing following command:
```
//create venv
python -m venv .venv

//activate .venv
source .venv/Scripts/activate

//upgrade pip
python -m pip install --upgrade pip

//instal package in editable mode
python -m pip install -e .

//clean egg-info artifact
python setup.py clean
```

```
kaggle competitions download -c child-mind-institute-problematic-internet-use -p data/bronze_dataset

```

Then unzip inside folder original_data

# How To
TO DO