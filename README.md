# Book Recommendation System Using Collaborative Filtering

A Python-based recommender system built on the Book-Crossing dataset. Utilizes matrix factorization (SVD and NMF) via the Surprise library to generate personalized book recommendations. Includes exploratory data analysis (EDA), model training, hyperparameter tuning, and error analysis.

---

## Features

### Matrix Factorization Models
- SVD and NMF implementations using Surprise  
- Cross-validation to compare model performance (RMSE, MAE, fit/test time)  
- GridSearchCV for SVD hyperparameter tuning (n_factors, n_epochs, lr_all, reg_all)  

### Exploratory Data Analysis (EDA)
- Distribution of ratings per user and per book  
- Rating distribution by publication year  
- Top authors and top-rated books analysis  
- Correlation between explicit (1–10) and implicit (click) ratings  
- Demographic breakdown of users (age, country)  

### Personalized Recommendations
- Train/test split evaluation  
- Prediction error distribution and analysis by actual rating  
- Generation of Top-N book recommendations for sample users  
- Comparison of predicted vs. actual preferences  

---

## Tech Stack

### Programming & Data
- Python  
- Pandas  
- NumPy  

### Visualization
- Matplotlib  
- Seaborn  
- PyWaffle (country share waffle chart)  

### Recommender Engine
- Surprise (SVD, NMF, cross_validate, train_test_split, GridSearchCV)  
