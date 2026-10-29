# Dashboard Quick Start

```
# from the project folder, with the virtual environment active
pip install "pyarrow==25.0.1"     # only needed once
python main.py --train            # only needed once (and after changing training code)
cd dashboard
streamlit run app.py
```

Open http://localhost:8501. The sidebar should say **"All trained models found"**.

Which file to upload on which page: see the table in `README.md`. Sample files are in `data/samples/`.
