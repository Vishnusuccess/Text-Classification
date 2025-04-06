
## Requirements

To set up the environment, install the necessary dependencies listed in `requirements.txt`. This project is compatible with Python 3.10.5.


### Python Version

- Python 3.10.5

### Setting up the environment

1. Create a virtual environment:

   ```bash
   python3 -m venv .venv
   ```
2. Start the environment for MacOS
    ```bash
    source .venv/bin/activate
    ```
    For Windows 
    ```bash
    .\.venv\Scripts\activate
    ```

### Install Dependencies 

```bash
pip install -r requirements.txt
 ```

### Instal Spacy English Model 
```bash
python -m spacy download en_core_web_sm
 ```
### Model Files
XGBoost and LightGBM models are trained in model.py. The trained models are saved in lgbm_model.pkl and xgb_model.pkl respectively.

Deep learning model (MLP) is trained in deepmodel.py 

The models are loaded in app.py to make predictions based on the description input.



## How to Use the API
### Start the Flask server:
```bash
python app.py
 ```
 This will start the Flask server on http://127.0.0.1:5000/ by default.

 Then go to the http://127.0.0.1:5000/docs give the description and do the Prediction.
