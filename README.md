# Dementia Text Classification Project

This project aims to classify text abstracts related to different types of dementia, including: 
* Alzheimer's Disease
* Frontotemporal Dementia
* Lewy Body Dementia
* Parkinson's Disease
* Vascular Dementia

The project employs various Natural Language Processing (NLP) techniques and Machine Learning models ranging from traditional Bag-of-Words models to modern Transformer-based architectures like BERT.

## Project Structure

The workspace is organized into data processing pipelines, analysis notebooks, and model implementations.

### Scripts and data files
*   **`1. data_extracting.ipynb`**: The initial data pipeline. It loads raw CSV datasets from the `datasets/` folder, samples documents, performs initial preprocessing (chunking), and consolidates them into `processed_data.csv`.
<<<<<<< HEAD
*   **`2. data_cleaning.ipynb`**: Performs advanced text cleaning. It normalizes text, removes stopwords, and crucially masks disease-specific terms (like "Alzheimer", "Dementia") to prevent data leakage during classification. Take in `processed_data.csv` and produces `cleaned_data.csv`.
=======
*   **`2. data_cleaning.ipynb`**: Performs advanced text cleaning. It normalizes text, removes stopwords, and crucially masks disease-specific terms (like "Alzheimer", "Dementia") to prevent data leakage during classification. Produces `cleaned_data.csv`.
>>>>>>> 554137625baa8d116216dc685a48b04d33f177cd
*   **`processed_data.csv`**: The successfully abstracted dataset from raw datasets used for training models.
*   **`cleaned_data.csv`**: The final preprocessed dataset used for training models.

### Directories
*   **`analysis/`**: Contains notebooks for comparisons of hardship experiments and baseline models.
    *   Example: `CountVectorizer (BoW) MultinomialNB.ipynb`, `TF-IDF + SGD.ipynb`.
*   **`models/`**: Dedicated notebooks for training, validating, and interpreting specific models.
    *   **Models**: `Random Forest + N-gram.ipynb`, `XGBoost + N-gram.ipynb`, `TF-IDF + SVM.ipynb`, etc.
    *   **Deep Learning**: `BERT_Classification.ipynb` (Fine-tuning BERT).
  
*   **`datasets/`**: Source data folder containing the original CSV files for each disease class.

## Environment & Requirements

This project works with Python 3.12+ and requires the following major libraries:

*   **Data Processing**: `pandas`, `numpy`
*   **Visualization**: `matplotlib`, `seaborn`
*   **Machine Learning**: `scikit-learn`, `xgboost`
*   **Deep Learning (BERT)**: `torch`, `transformers`
*   **NLP Tools**: `nltk`
*   **Interpretability**: `shap`

### Installation

To set up the environment, install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost torch transformers nltk shap
```

## Getting Started

1.  **Data Preparation**: Ensure your raw CSV files are in the `datasets/` folder.
2.  **Preprocessing**: Run `1. data_extracting.ipynb` to generate `processed_data.csv` followed by `2. data_cleaning.ipynb` to generate `cleaned_data.csv`.
3.  **Modeling**: Open any notebook in `models/` (e.g., `XGBoost + N-gram.ipynb`) to train a model and view performance metrics (Confusion Matrix, ROC Curve) and SHAP explanations.
