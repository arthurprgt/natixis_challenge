# natixis_challenge

Create virtual environment
```bash
conda create --name natixis python=3.9
conda activate natixis
```

Install requirements
```bash
pip install -r requirements.txt
```

To install the local package run
```bash
pip install -e .
```

1. Add your data as *data.csv* to the data folder
2. To create the dataset used to train the model run
```bash
python src/natixis/deep_model/create_data.py
```
3. To run the app run
```bash
streamlit run app/app.py
```
4. To retrain the model run
```bash
python main.py
```
