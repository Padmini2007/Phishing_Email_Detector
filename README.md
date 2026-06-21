
# 🛡️ Phishing Email Detection Model

A machine learning project built with **Python** and **Scikit-learn** that classifies emails as **Phishing** or **Safe**, combining TF-IDF text analysis with engineered security features (URL count, suspicious keywords, IP-based links, etc.). Includes a `RandomForestClassifier`, full evaluation metrics, and an interactive **Streamlit** web app.

---

## DEMO_VIDEO_LINK : https://drive.google.com/file/d/1WMLvWweeq01rDiGy4SwfvVSOn8tzet-4/view?usp=drivesdk

---

## ✨ Features
- RandomForest classifier trained on TF-IDF + handcrafted features
- Detects suspicious URLs, IP-based links, and urgency/scare keywords
- Reports Accuracy, Precision, Recall, F1-score, Confusion Matrix, and ROC Curve
- Interactive Streamlit web app with live confidence scores
- Saved model artifacts (`.joblib`) for instant reuse

---

## 🛠️ Tech Stack
Python · Scikit-learn · Pandas · NumPy · Matplotlib/Seaborn · Streamlit · Plotly

---

## 📁 Project Structure

phishing_detector/

├── generate_dataset.py      # Creates the labeled email dataset

├── feature_engineering.py   # Extracts URL/keyword/structural features

├── phishing_detector.py     # Trains, evaluates, and saves the model

├── app.py                   # Streamlit web app

├── requirements.txt

└── README.md

---

## ⚙️ Installation
```bash
git clone https://github.com/<your-username>/phishing-email-detector.git
cd phishing-email-detector
pip install -r requirements.txt
```

---

## ▶️ Usage
```bash
python generate_dataset.py      # creates emails_dataset.csv
python phishing_detector.py     # trains, evaluates, saves model
python -m streamlit run app.py  # launches the web app
```

---

## 📈 Model Performance
On the included dataset (600 emails, 80/20 split): **~100% accuracy**.
> Note: the bundled dataset is synthetically generated for demo purposes — use a real-world dataset (e.g. from Kaggle) for a realistic benchmark.

---

## 👩‍💻 Author
**Developed by PADMINI J**

## 📄 License
MIT License — free to use, modify, and distribute with attribution.
