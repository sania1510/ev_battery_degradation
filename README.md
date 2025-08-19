
# 🔋 EV Battery Degradation Risk Index (RRI) Prediction

This project predicts **EV battery health degradation** by combining three factors:  

- **EV usage conditions** (distance, speed, charging, temperature)  
- **Traffic stress** (traffic density, time-of-day, junctions)  
- **Weather stress** (temperature, humidity, precipitation, visibility)  

We integrate these signals into a **Relative Risk Index (RRI)** to show how scenarios like **“High Traffic + Cloudy Weather”** vs. **“Low Traffic + Clear Weather”** affect battery degradation.

---

## 🚀 Features

- ✅ **Machine Learning Models** trained on EV, traffic, and weather datasets  
- ✅ **Comparison of Models (Linear Regression, Random Forest, XGBoost)** with performance metrics (MAE, RMSE, R²)  
- ✅ **Streamlit Dashboard** for interactive exploration  
- ✅ **Relative Risk Index (RRI)** combining EV (60%), Traffic (25%), and Weather (15%) stress signals  
- ✅ **Visualizations with Seaborn**:  
  - Correlation matrix  
  - Distribution plots  
  - RRI comparisons across scenarios  
  - **Trend line plots** (RRI vs Temperature / Traffic Density)  

---

## 📊 Datasets

1. **EV Dataset**  
   - total_distance_km, average_trip_speed_kmph, ambient_temperature_C,  
     charging_cycles, fast_charging_ratio_%, average_battery_temperature_C, battery_health_%  

2. **Traffic Dataset**  
   - DateTime, Junction, Vehicles, ID  

3. **Weather Dataset**  
   - DateTime, Summary, Precip Type, Temperature (C), Humidity, Wind Speed, Visibility, Pressure, etc.  

---

## ⚙️ Tech Stack

- **Language**: Python 3.9+  
- **Libraries**:  
  - `scikit-learn` → Model training  
  - `xgboost` → Gradient boosting model  
  - `seaborn` + `pandas` → Data visualization & analysis  
  - `streamlit` → Interactive web app  

---

## 📂 Project Structure

```

📦 EV-Battery-Degradation
├── 📁 data                # Raw datasets (EV, Traffic, Weather)
├── 📁 results             # Model metrics & plots
│   ├── LinearRegression.json
│   ├── RandomForest.json
│   ├── XGBoost.json
│   └── ...
├── 📁 models              # Trained models (pickle files)
├── app.py                 # Streamlit app
├── model\_training.py      # Training + evaluation scripts
├── requirements.txt       # Dependencies
└── README.md              # Project documentation

````

---

## 🖥️ Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/sania1510/ev-battery-degradation.git
   cd ev-battery-degradation
````

2. **Create Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Mac/Linux
   venv\Scripts\activate      # On Windows
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run Streamlit App**

   ```bash
   streamlit run app.py
   ```

---

## 📈 Example Outputs

### 🔹 Model Performance

| Model             | MAE   | RMSE  | R²   |
| ----------------- | ----- | ----- | ---- |
| Linear Regression | 0.053 | 0.067 | 0.95 |
| Random Forest     | 0.018 | 0.027 | 0.99 |
| XGBoost           | 0.019 | 0.027 | 0.99 |

### 🔹 Visualizations

* **RRI Comparison by Scenario**
* **Distribution of Battery Health Degradation**
* **Trend Line of RRI vs Traffic Density**
* **Trend Line of RRI vs Temperature**

---

## 🎯 Future Enhancements

* 🔹 Integrate **real-time traffic + weather API data**
* 🔹 Add **deep learning models (TabNet / Transformers)**
* 🔹 Build **recommendation engine** for optimal driving & charging patterns
* 🔹 Deploy to **Streamlit Cloud**

---

## 🤝 Contribution

Contributions are welcome!

1. Fork the repo
2. Create a feature branch
3. Commit changes & push
4. Submit a PR 🚀

---

## 📜 License

This project is licensed under the **MIT License** – feel free to use and modify with attribution.

---

## 👨‍💻 Author

Built with ❤️ by **\[Your Name]**
📩 Connect on [LinkedIn](https://www.linkedin.com/in/sania-6a9561292) | 🌐 [GitHub](https://github.com/sania1510)


