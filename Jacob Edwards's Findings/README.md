# Jacob Edwards's Findings — Stocks

- **Model (Hugging Face):** https://huggingface.co/jacobre20/stock-sentiment-daily-v1  
- **Metrics (holdout):** ACC ≈ 0.52, AUC ≈ 0.523, F1 ≈ 0.68  
- **Notes:** Daily direction prediction is noisy; model slightly beats “Always Up.”  
- **Next:** daily pipeline + weekly batch retrain.

See the notebook in `Stocks/stock_model_colab.ipynb`.
