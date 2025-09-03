# Week 4: Deploy a Toy ML Model on Flask

This repo demonstrates the full pipeline your assignment asks for:
1. **Select toy data** → Iris dataset from `sklearn.datasets`  
2. **Train & save model** → `train_model.py` saves a `models/iris_pipeline.joblib`  
3. **Deploy on Flask** → `app.py` serves a simple web app form + `/predict`  
4. **Create PDF document** → Use `DOCS/Submission_Document.md` as a template; paste screenshots and export to PDF  
5. **Upload to GitHub** → Push this folder to a new GitHub repo  
6. **Submit URL** → Submit link to the PDF file in your GitHub

---

## Quickstart

```bash
# 1) Create and activate a virtual env (recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install requirements
pip install -r requirements.txt

# 3) Train & save the model
python train_model.py

# 4) Launch the Flask app
python app.py
# Open http://127.0.0.1:5000/ in your browser
```

## What to Screenshot for Your PDF

Paste the following screenshots into `DOCS/Submission_Document.md`, then export to PDF:
- **Toy Data**: show the Iris dataset description or a small preview (`train_model.py` output).  
- **Model Saved**: show terminal with `Saved pipeline to models/iris_pipeline.joblib`.  
- **Flask Running**: screenshot your terminal `* Running on http://127.0.0.1:5000` and the web form in your browser.  
- **Prediction**: enter sample values and screenshot the predicted class + probabilities.  
- **GitHub**: screenshot your GitHub repo page with the PDF present.

Then convert `DOCS/Submission_Document.md` to PDF (open it in VS Code + Markdown PDF extension, paste into Google Docs and *File → Download → PDF*, etc.).

## Deploy Notes

- For local demo, `python app.py` is enough.
- For hosting (optional), use gunicorn: `gunicorn -w 2 -b 0.0.0.0:8000 app:app` (on a server/Render/railway/etc.).

## Repo Layout

```
flask_iris_deploy/
├── app.py
├── train_model.py
├── requirements.txt
├── models/
│   └── iris_pipeline.joblib  # (created after training)
├── templates/
│   └── index.html
├── static/
├── DOCS/
│   └── Submission_Document.md
└── README.md
```

Good luck! 🎉
