# House Price Prediction using TensorFlow Decision Forests

## Description:
This project focuses on predicting house prices based on various features using **TensorFlow Decision Forests (TF-DF)**. The dataset consists of features such as the size of the house, number of rooms, neighborhood, and other related factors that impact the house price.

The model uses a **Random Forest** algorithm, a tree-based ensemble learning method, to predict the **SalePrice** of the house. TF-DF supports all feature types (numeric, categorical, and missing values) natively, allowing us to work with raw data without the need for extensive preprocessing.

## Key Features:
- **Random Forest Model**: A collection of decision trees, each trained on a random subset of the data, providing robust and accurate predictions.
- **Feature Importance**: The model ranks the features based on their contribution to the final prediction, helping us understand which features are the most influential in determining house prices.
- **Out-of-Bag Evaluation**: The model utilizes OOB (Out-of-Bag) samples for evaluation, ensuring robust performance metrics.
- **Model Evaluation**: The model is evaluated using RMSE (Root Mean Squared Error), ensuring that predictions are accurate.
- **Predictions on Test Set**: After training, predictions are made on the test dataset and saved in a CSV file for further analysis or submission.

## Project Workflow:
1. **Data Preprocessing**: Data is cleaned, missing values are handled, and features are converted into a format suitable for training the model.
2. **Model Training**: The Random Forest model is trained using TensorFlow Decision Forests on the training data.
3. **Model Evaluation**: The model’s performance is evaluated using Out-of-Bag (OOB) error and RMSE metrics.
4. **Predictions**: The trained model is used to predict house prices on a test dataset, and results are saved in a CSV file.

## Technologies Used:
- **TensorFlow Decision Forests (TF-DF)**: A framework for training decision forests (Random Forest, Gradient Boosted Trees, etc.) in TensorFlow.
- **Pandas**: For data manipulation and processing.
- **Matplotlib / Seaborn**: For visualizing data and model evaluation results.
- **NumPy**: For numerical operations.
- **Google Colab**: Used for training the model and running the code.

## Files in the Repository:
- `house_prices_prediction.ipynb`: The main Jupyter notebook containing the code for data processing, model training, evaluation, and prediction.
- `test.csv`: The test dataset used to make predictions.
- `predictions.csv`: The output CSV file containing predictions of the house prices.
