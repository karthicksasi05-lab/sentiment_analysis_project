# Financial News Sentiment Prediction (Deep Learning)

Classifies finance tweets / headlines as **Bearish**, **Bullish** or **Neutral**.

- **Main project:** Embedding + Simple RNN, Embedding + LSTM, Embedding + GRU (PyTorch)
- **Optional extra:** fine-tuned FinBERT, trained separately on Google Colab
- **Dataset:** `zeroshot/twitter-financial-news-sentiment` (Hugging Face)
- **Metric:** accuracy and macro F1 (the data is imbalanced, Neutral is the biggest class)

## Project files

| File | What it is |
|------|-----------|
| `Financial_Sentiment_Analysis_Final.ipynb` | **Main notebook** (no BERT): data, cleaning, EDA, RNN/LSTM/GRU training, comparison. Every cell has an explanation. |
| `FinBERT_Colab.ipynb` | **Separate notebook** to fine-tune FinBERT on Google Colab (GPU) and compare it with the RNN models |
| `utils.py` | Shared functions (cleaning, model class, loading, prediction) |
| `app.py` | Streamlit dashboard |
| `predict.py` | Command line script: paragraph in, sentiment out |
| `requirements.txt` | Python packages |
| `data/` | Created by the notebooks (CSV files, charts, `model_comparison.csv`) |
| `models/` | Created by the notebooks (saved models) |

## Step 1: main notebook (RNN / LSTM / GRU)

**On Google Colab:** upload `Financial_Sentiment_Analysis_Final.ipynb`, choose Runtime > Change runtime type > T4 GPU
(optional, faster), then Runtime > Run all. The last section downloads `saved_results.zip`. Unzip it into the
project folder.

**On your own computer (VS Code / Jupyter):**

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac / Linux
pip install -r requirements.txt
```

Open the project folder in VS Code, open the notebook, select the `venv` kernel and Run All.
This creates `data/` and `models/` next to the notebook.

## Step 2 (optional): FinBERT on Google Colab

1. Upload `FinBERT_Colab.ipynb` to Colab.
2. Runtime > Change runtime type > **T4 GPU** (needed, on a CPU it takes hours).
3. Optional: upload `data/model_comparison.csv` from step 1 to the Colab file panel, so the notebook can compare
   FinBERT with RNN / LSTM / GRU.
4. Runtime > Run all (about 5 to 10 minutes). The last cell downloads `finbert_results.zip`.
5. Unzip it into the project folder so that `models/finbert_sentiment/` and `data/bert_results.json` are in place.
6. On your computer run `pip install transformers` once.

## Run the dashboard

```bash
streamlit run app.py
```

Needs `models/` and `data/` next to `app.py`. The sidebar shows the RNN/LSTM/GRU model. If you finished step 2,
**FinBERT** also appears as a second choice and is added to the comparison table.

## Predict from the command line

```bash
python predict.py "Shares surge after record profit. Analysts raise price targets."
python predict.py --file news.txt
python predict.py --model bert "Company misses estimates and cuts guidance."     # only after step 2
```

Run the commands from the project folder.

## Results

Fill in from `data/model_comparison.csv` (and `data/bert_results.json` if you did step 2):

| Model | Val Accuracy | Val Macro F1 |
|-------|--------------|--------------|
| Simple RNN | | |
| LSTM | | |
| GRU | | |
| FinBERT (optional) | | |

## Notes

- Train and validation splits are the official ones from Hugging Face (9,543 / 2,388 rows in the
  current version of the dataset; the project brief mentions 9,938 / 2,486).
- The vocabulary is built from the training data only, so there is no data leakage.
- Early stopping on validation macro F1 and gradient clipping are used to keep the RNN training stable.
