# Intelligent Resume Screening System

## Project Overview

The Intelligent Resume Screening System is an AI-based application designed to help recruiters screen and rank resumes according to their relevance to a given job description.

The system uses Natural Language Processing (NLP), TF-IDF feature extraction, cosine similarity, skill matching, and machine learning classification to analyze resumes and identify suitable candidates.

## Problem Statement

Recruiters often spend a significant amount of time manually reviewing large numbers of resumes. This process can be time-consuming and may make it difficult to identify the most relevant candidates quickly.

This project aims to automate the initial resume screening process by analyzing resume content and ranking candidates based on their relevance to a job description.

## Objectives

- Extract and preprocess resume text.
- Remove unnecessary words and characters using NLP techniques.
- Convert resume text into numerical features using TF-IDF.
- Compare resumes with job descriptions using cosine similarity.
- Identify required skills present in each resume.
- Calculate an overall candidate score.
- Rank resumes according to suitability.
- Classify resumes into different professional categories.
- Provide an interactive Streamlit interface for resume screening.

## Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Matplotlib
- Streamlit
- Jupyter Notebook

## Machine Learning Techniques

### 1. Text Preprocessing

Resume text is converted to lowercase and cleaned by removing URLs, special characters, stop words, and very short words.

### 2. TF-IDF

TF-IDF is used to convert resume text into numerical feature vectors.

The project uses:

- Maximum features: 5000
- Unigrams and bigrams

### 3. Cosine Similarity

Cosine similarity is used to measure the relevance between a job description and each resume.

### 4. Skill Matching

The system checks whether important skills required for the job are present in the candidate's resume.

### 5. Candidate Scoring

The final candidate score is calculated using:

- 60% Resume Relevance
- 40% Skill Match

### 6. Classification

A Logistic Regression model is trained to classify resumes into professional categories.

## Dataset

The project uses the Kaggle Resume Dataset.

Dataset source:

https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

The dataset contains resume text and professional categories.

## Model Performance

The Logistic Regression classification model achieved:

**Accuracy: 66.40%**

The model was trained using 80% of the dataset and tested using 20%.

## Application Features

The Streamlit application provides:

- Job description input
- Candidate resume input
- Resume relevance score
- Skill match percentage
- Final candidate score
- Matched skills
- Resume category prediction
- Candidate suitability status
- Modern AI-style interface

## Project Structure

```text
Intelligent-Resume-Screening-System/
│
├── app.py
├── README.md
├── .gitignore
│
├── dataset/
│
├── notebooks/
│   └── resume_screening.ipynb
│
├── results/
│   ├── classifier.pkl
│   ├── ranked_resumes.csv
│   ├── required_skills.pkl
│   └── tfidf_vectorizer.pkl
│
├── screenshots/
│
├── reports/
│
└── src/
## Application Screenshot

![Intelligent Resume Screening System](screenshots/resume_screening.png)
