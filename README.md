# 💼 Data Science Salary Prediction App

This is a professional, interactive Streamlit app that predicts data science salaries around the world. Enter your job role, experience, country, and working style to estimate your expected salary, powered by the latest industry datasets and machine learning models.

## 🚀 Features

- **Accurate predictions:** Trained on the Data Science Salaries 2023 Kaggle dataset.
- **Modern interface:** Dark theme, full-form dropdowns, and clear country selection.
- **Privacy-friendly:** No user data stored.
- **Fully customizable:** Easily retrain or swap models.
- **Live insights:** Predictions update instantly.

## 🛠️ How It Works

1. Fill out your profile: job role, experience, company size, work country, and remote preferences.
2. Click "Predict Salary" to get your personalized annual salary estimate in USD.
3. Adjust your profile to explore how experience or location affects your earning power.

## 🏗 Quickstart

1. **Clone or download this repository.**
2. (Optional) Train your own model using the code and data scripts.
3. Ensure Python 3.8+ and required packages are installed:

    ```
    pip install -r requirements.txt
    ```

4. Start the app:

    ```
    streamlit run app.py
    ```

5. Open `http://localhost:8501` in your browser.

## 📂 Files

| File                  | Purpose                               |
|-----------------------|---------------------------------------|
| `app.py`              | Main Streamlit application            |
| `salary_model.pkl`    | Trained machine learning model        |
| `preprocessor.pkl`    | Encoders for categorical fields       |
| `column_mappings.pkl` | Maps raw data columns to model input  |
| `model_info.pkl`      | Model’s feature configuration         |
| `requirements.txt`    | List of required Python libraries     |
| Other scripts         | For data cleaning/training, optional  |

## ⚠️ .gitignore

A `.gitignore` file is included to keep data and model binaries out of the repository by default.  
If you want to share models or datasets, remove those lines.


## 📊 Data Source

Original salary data: [Kaggle - Data Science Salaries 2023](https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023)

## 📝 License

Open source for learning and demo use.  
Please credit the authors and data sources if you share or build upon this work.

---

**Built with ❤️ using Streamlit, Pandas, scikit-learn, and open salary data.**

