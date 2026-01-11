# 🌾 Crop Recommendation System

**Intelligent agricultural platform that predicts the most suitable crop for given soil and weather conditions using machine learning.**  
Built with a Flask backend and responsive frontend to help users make data-driven decisions for better yields.

---

## 📌 Overview

This project uses soil nutrient and environmental parameters to recommend the optimal crop to cultivate.  
The model considers the following features:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- pH
- Rainfall

The underlying model achieves **99%+ accuracy** across multiple crop classes using supervised machine learning classifiers. :contentReference[oaicite:1]{index=1}

---

## 🧠 Features

- 🧪 Machine learning classification for crop suitability  
- 📈 High prediction accuracy (>99%)  
- 🌐 Flask backend serving predictions  
- 📱 Responsive simple frontend UI  
- 📦 Pretrained models included (pickle files)  
- 🧾 Label encoding and scaling included for production inference

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-------------|
| Backend | Python, Flask |
| Model Storage | Pickle (`.pkl`) |
| ML Libraries | scikit-learn, ensemble models |
| Frontend | HTML / CSS |

---

## 📁 Repository Contents

- `crop.ipynb` — Notebook containing training & evaluation code  
- `best_crop_model.pkl` — Pretrained model used for inference  
- `label_encoder.pkl` — Stores label encoding mapping  
- `scaler.pkl` — Feature scaler for preprocessing  
- `templates/` — HTML templates for the UI  
- `README.md` — This documentation

---

## 🚀 Installation

1. **Clone the repository**

```
git clone https://github.com/h4ck3r0/crop_recommendation
cd crop_recommendation
````

2. **Install dependencies**

```
pip install -r requirements.txt
```

3. **Run the Flask app**

```
python app.py
```

4. **Open your browser**

Visit: `http://127.0.0.1:5000`

---

## 📊 How It Works (Backend Logic)

1. Load dataset and preprocess features
2. Apply label encoding to target crop labels
3. Train multiple ML classifiers
4. Select the best performing model
5. Serve model predictions via the Flask API

The model compares soil and weather inputs against learned patterns to provide the most suitable crop outcome.

---

## 📈 Model Performance

Model performance is evaluated using metrics like **accuracy**, **confusion matrix**, and **cross-validation** to ensure reliability across 22 crop categories. This aligns with common practices in crop recommendation research and applications. ([GitHub][1])

---

## 🧠 Example Usage

Once the Flask server is running, you can input soil and weather values in the UI form to get a recommended crop.

---

## 👤 Author

**Raj Aryan (H4CK3R)**

* GitHub: [https://github.com/h4ck3r0](https://github.com/h4ck3r0)
* Email: [rajaryan2315@gmail.com](mailto:rajaryan2315@gmail.com)

---

## 📜 License

This project is licensed under **GPL-3.0 License**. ([GitHub][2])

---

## 📌 Citation

If you use this project for research or coursework, please cite it appropriately.

---

