# Prediction to Numerai Models

This is an application that uses ML models to compete in Numerai tournaments. The scripts are written and intended to use with Python 3.

## Getting Started

### 1. Clone Repo

`git clone https://github.com/andrebrener/numerai_models.git`

### 2. Install Packages Required

Go in the directory of the repo and run:
```pip install -r requirements.txt```

### 3. Download Numerai Datasets
- Download the datasets from [Numerai](https://numer.ai/)
- Save these files inside a directory called `numerai_datasets` inside the repo.

### 4. Insert Constants

In constants.py enter the following parameters:

- Parameters for each model to perform the predictions. Remember the script will pick the best parameters for the best performing model.
- The percentage categorization. The script categorizes each variable by a percentile of the values in it.

### 5. Run model

- Run [main.py](https://github.com/andrebrener/numerai_models/blob/master/main.py).
- When the script finishes, a directory for each categorization will be created inside `numerai_datasets`. Inside this folder, you will be able to see the training and tournament tables, and the final csv file with the target predictions to be uploaded directly.
