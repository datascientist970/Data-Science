# Data Science Projects

> A portfolio collection of end-to-end data science, machine learning, deep learning, NLP, and computer vision projects — built with Python and Jupyter Notebooks.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?logo=tensorflow&logoColor=white)
![Django](https://img.shields.io/badge/Django-Web%20App-092E20?logo=django&logoColor=white)

## Overview

This repository is a growing collection of hands-on data science projects covering the full workflow: data understanding, cleaning and preprocessing, exploratory analysis, feature engineering, model building, evaluation, and insight generation. It also includes a Django web application (`ai_agent/`) that serves trained health-prediction models alongside a LangChain / Gemini AI agent.

## Project Index

Every distinct project bundled in this repository:

| # | Project | What it does | File / Folder |
|---|---------|--------------|----------------|
| 1 | Predictive System | General predictive modelling workflow in a notebook | [`predictive System.ipynb`](predictive%20System.ipynb) |
| 2 | Anomaly Detection | Detects anomalous patterns in data | [`Anomaly_detection.ipynb`](Anomaly_detection.ipynb) |
| 3 | Brain Tumor Classification | Notebook exploring brain-tumor prediction/classification | [`Brain_tumor.ipynb`](Brain_tumor.ipynb) |
| 4 | Car Price Prediction | Regression model that predicts car prices | [`Car Price prediction .ipynb`](Car%20Price%20prediction%20.ipynb) |
| 5 | Leaf Disease Prediction | Plant leaf disease prediction from data/images | [`Leaf_desease_prediction.ipynb`](Leaf_desease_prediction.ipynb) |
| 6 | Tesla (TSLA) Stock EDA | Exploratory data analysis of Tesla stock data | [`TSLA_EDA.ipynb`](TSLA_EDA.ipynb) |
| 7 | Book Recommendation | Book recommendation system | [`book_recommendation.ipynb`](book_recommendation.ipynb) |
| 8 | Cancer Prediction | Cancer prediction modelling notebook | [`cancer.ipynb`](cancer.ipynb) |
| 9 | Customer Segmentation | Customer clustering/segmentation analysis | [`customer Segmentation.ipynb`](customer%20Segmentation.ipynb) |
| 10 | Heart Failure Prediction | Predicts heart failure risk | [`heart_failure.ipynb`](heart_failure.ipynb) |
| 11 | House Price Prediction | Regression model for house prices | [`house price.ipynb`](house%20price.ipynb) |
| 12 | Image Classification | Introductory image-classification notebook | [`image.ipynb`](image.ipynb) |
| 13 | Iris Classification | Classic Iris dataset classification | [`iris.ipynb`](iris.ipynb) |
| 14 | Movie Recommendation | Movie recommender notebook | [`movie.ipynb`](movie.ipynb) |
| 15 | Sentiment Analysis | Text sentiment classification | [`sentiment_analysis.ipynb`](sentiment_analysis.ipynb) |
| 16 | Face Detection | Real-time face detection from video using OpenCV Haar cascades | [`face_detection.py`](face_detection.py) |
| 17 | Object / Motion Detection | Motion-based object detection and counting in video with OpenCV | [`object_detection.py`](object_detection.py) |
| 18 | Perspective Warp Demo | Perspective transformation of an image region with OpenCV | [`wrap.py`](wrap.py) |
| 19 | AI Health Agent (Django App) | Django web app with pages for cancer, diabetes and heart prediction using saved models (`cancer.pkl`, `heart.pkl`, `diabetes_model.h5`), plus a LangChain ReAct agent powered by Google Gemini (`gemini-2.0-flash`) with DuckDuckGo search | [`ai_agent/`](ai_agent/) |
| 20 | BioTrack Attendance System (archive) | Archived project bundle distributed as a ZIP file in this repository | [`BIO_TRACK_ATTENDENCE.zip`](BIO_TRACK_ATTENDENCE.zip) |

## Tech Stack

- **Language:** Python
- **Data & ML:** Pandas, NumPy, scikit-learn, Matplotlib, Seaborn
- **Deep Learning:** TensorFlow / Keras (e.g. the diabetes model in `ai_agent`)
- **Computer Vision:** OpenCV
- **NLP / Agents:** LangChain, Google Gemini, DuckDuckGo Search
- **Web:** Django (in `ai_agent/`)
- **Environment:** Jupyter Notebook, Git & GitHub

## Project Structure

```text
Data-Science/
├── *.ipynb                  # Standalone ML / DL / EDA notebooks (see Project Index)
├── face_detection.py        # OpenCV face detection from video
├── object_detection.py      # OpenCV motion / object detection
├── wrap.py                  # OpenCV perspective transform demo
├── BIO_TRACK_ATTENDENCE.zip # Archived attendance-system project bundle
└── ai_agent/                # Django app: health predictions + LangChain/Gemini agent
    ├── agent/               # Views, templates and saved models (pkl / h5)
    ├── ai_agent/            # Django project settings
    └── manage.py
```

## Getting Started

```bash
git clone https://github.com/datascientist970/Data-Science.git
cd Data-Science
pip install pandas numpy scikit-learn matplotlib seaborn opencv-python tensorflow jupyter
jupyter notebook
```

Open any notebook from the Project Index and run the cells top to bottom.

### Running the Django AI Health Agent

```bash
cd ai_agent
pip install django langchain langchain-community tensorflow duckduckgo-search
python manage.py runserver
```

> The agent view expects a Google Gemini API key — replace the placeholder key in `ai_agent/agent/views.py` with your own before use.

## Usage

- **Learners:** work through the notebooks in order — Iris and House Price for fundamentals, then segmentation, recommendation and computer-vision projects.
- **Recruiters / collaborators:** the Project Index above is the fastest way to jump to a specific skill area.

## Author

**Muhammad Dawood** — AI & Data Science Engineer
GitHub: https://github.com/datascientist970 · Mardan, Pakistan
LinkedIn: https://www.linkedin.com/in/dawood406/

⭐ If you find this repository useful, consider giving it a star.
