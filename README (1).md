# 🌸 Iris Flower Classification
### AICTE OASIS Data Science Internship — Task 1

---

## 📌 Project Overview

This project trains a **Machine Learning model** to classify Iris flowers into 3 species:
- 🌺 **Setosa**
- 🌼 **Versicolor**
- 🌸 **Virginica**

The model learns from 4 measurements of each flower:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

---

## 📁 Folder Structure

```
your_folder/
│
├── iris_flower_classification.py   ← Main Python code
├── iris.csv                        ← Dataset file
├── README.md                       ← This file
│
└── (These are auto-generated after running)
    ├── iris_pairplot.png
    ├── iris_boxplot.png
    ├── iris_confusion_matrix.png
    └── iris_feature_importance.png
```

---

## 🛠️ Requirements

Install the required libraries by running this command once in your terminal:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## ▶️ How to Run

```bash
python iris_flower_classification.py
```

Make sure `iris.csv` and the `.py` file are in the **same folder**.

---

## 📊 What the Code Does — Step by Step

| Step | Description |
|------|-------------|
| 1 | Install libraries |
| 2 | Import libraries |
| 3 | Load `iris.csv` dataset |
| 4 | Auto-detect feature & target columns |
| 5 | Clean data (remove nulls, encode labels) |
| 6 | Explore data (statistics, species count) |
| 7 | Visualize data (pairplot, box plots) |
| 8 | Split data — 80% train / 20% test |
| 9 | Train **Random Forest** model |
| 10 | Evaluate model (accuracy, report) |
| 11 | Plot feature importance |
| 12 | Predict a new flower 🌸 |

---

## 🤖 Machine Learning Model

- **Algorithm**: Random Forest Classifier
- **Trees**: 100 decision trees
- **Train/Test Split**: 80% / 20%
- **Expected Accuracy**: ~96–100%

---

## 📈 Output Graphs

| File | Description |
|------|-------------|
| `iris_pairplot.png` | Shows relationship between all features |
| `iris_boxplot.png` | Shows spread of measurements per species |
| `iris_confusion_matrix.png` | Shows correct vs wrong predictions |
| `iris_feature_importance.png` | Shows which feature matters most |

---

## 🌸 Sample Prediction

To predict your own flower, change these values in the code (Step 12):

```python
my_flower = [5.1, 3.5, 1.4, 0.2]
#            sepal_len, sepal_wid, petal_len, petal_wid
```

---

## 👤 Author

- **Internship**: AICTE OASIS Data Science Internship
- **Task**: Task 1 — Iris Flower Classification
- **Dataset**: [UCI Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)

---

## 📝 License

This project is made for educational purposes as part of the AICTE OASIS internship program.
