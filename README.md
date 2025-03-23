# Medical-ML

Medical-ML is a comprehensive machine learning repository designed for medical data analysis. Initially, the repository includes a pipeline for hypercoagulation analysis, but it is structured to easily integrate additional analyses and factors over time.

## Overview

Medical-ML is built with the following objectives:
- **Modular Design:** The code is organized into clear modules for data loading, preprocessing, feature selection, model training, evaluation, and visualization.
- **Extensibility:** The project is designed to easily incorporate analyses for various medical factors beyond hypercoagulation.
- **Reproducibility:** It provides a complete machine learning pipeline, from data preprocessing to model evaluation, with options for parameter optimization and multiple evaluation metrics.

## Features

- **Data Preprocessing:** Handles missing values, performs feature selection based on correlation, and applies imputation techniques.
- **Data Splitting and Sampling:** Includes functions to split data into training, internal test, and external test sets along with SMOTE for imbalanced data.
- **Modeling:** Implements various machine learning models such as Logistic Regression, Random Forest, SVM, Decision Tree, K-Nearest Neighbors, Gradient Boosting, Neural Networks, Naive Bayes, AdaBoost, and XGBoost.
- **Parameter Optimization:** Uses GridSearchCV for hyperparameter tuning.
- **Evaluation and Visualization:** Provides metrics calculation with confidence intervals, ROC curves, Precision-Recall curves, calibration curves, and SHAP visualizations for model interpretability.
- **Extensible Pipeline:** The modular structure allows for easy expansion to incorporate additional medical factors and analyses.

## Getting Started

### Prerequisites

Ensure you have Python 3.7 or later installed. The project uses several popular packages; you can install them via pip:

```bash
pip install -r requirements.txt
```

The `requirements.txt` should include packages such as:
- pandas
- numpy
- scikit-learn
- matplotlib
- shap
- imbalanced-learn
- xgboost (optional, if using XGBoost)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your_username/Medical-ML.git
   cd Medical-ML
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

The repository is organized into modules. To run the complete machine learning pipeline for the current hypercoagulation analysis, execute:

```bash
python main.py
```

This will:
- Load and preprocess the data.
- Split the data into training, internal testing, and external testing sets.
- Apply sampling (SMOTE) to balance the dataset.
- Perform feature selection and standardization.
- Train and evaluate multiple machine learning models.
- Save evaluation results, plots, and trained models to the output directory.

## Project Structure

```
Medical-ML/
├── data/                  # Directory for input data and processed files
├── output/                # Directory for saving model outputs and visualizations
├── main.py                # Main execution script
├── modules/               # Directory containing individual modules (e.g., data processing, modeling)
├── README.md              # This file
└── requirements.txt       # List of project dependencies
```

## Contributing

Contributions are welcome! If you have ideas or improvements, feel free to:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes.
4. Push to the branch.
5. Open a pull request.

Please ensure that your code adheres to the existing coding style and includes appropriate documentation.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions or suggestions, please open an issue or contact the repository maintainer.
