## Airline Passenger Satisfaction Analysis

# Machine Learning Project — PCA + Logistic Regression

## Project Overview

Machine learning analysis of airline passenger satisfaction using PCA and Logistic Regression to identify key drivers of customer experience.

This project explores the key factors influencing airline passenger satisfaction using machine learning techniques. The goal was not only to build a predictive model but also to understand which aspects of the travel experience matter most to passengers.

The analysis combines data cleaning, feature engineering, dimensionality reduction (PCA), and classification modelling (Logistic Regression) to uncover patterns in passenger satisfaction.
## Research Questions

This project explores the following questions:

1. How strongly do flight delays influence passenger satisfaction?
2. Does service quality outweigh operational issues such as delays?
3. How does travel class affect satisfaction levels?
4. Do loyal customers respond differently to service experiences?
5. Can hidden service dimensions be discovered using PCA?

## Project Motivation

Airlines collect large amounts of feedback data from passengers, covering aspects such as:

seat comfort

onboard service

booking experience

flight delays

class of travel

However, understanding which factors truly drive satisfaction is not always straightforward.

The goal of this project was therefore to answer:

What factors have the strongest impact on airline passenger satisfaction?

## Dataset

The dataset contains passenger experience ratings and flight information including:

travel class

customer type

service quality metrics

flight delays

satisfaction label

The dataset includes `129,880` passenger records, making it large enough for meaningful statistical analysis and machine learning modelling.

## Approach

The project followed a structured machine learning workflow:

## Data exploration

Initial exploration revealed that Arrival Delay contained missing values.
Instead of dropping these rows, a Linear Regression model was used to estimate missing arrival delays based on related features.

This decision preserved valuable data and avoided unnecessary data loss.

## Feature engineering

Several transformations were applied:

Label Encoding for satisfaction target

One-hot encoding for travel class

PCA for service-related features

The service ratings contained 14 correlated variables, which could introduce multicollinearity.

To address this, Principal Component Analysis (PCA) was applied to compress these into key underlying service dimensions.

## PCA Interpretation

The PCA results revealed two major latent factors:

PC_1 — Overall onboard service quality

seat comfort

cleanliness

onboard service

entertainment

legroom

PC_2 — Pre-boarding experience

online booking

check-in service

gate location

boarding process

This dimensionality reduction simplified the model while preserving important information.

## Baseline model: Delay-only logistic regression

Initially, I built a logistic regression model using only:

Arrival Delay

Departure Delay

The goal was to test the hypothesis that flight delays strongly affect passenger satisfaction.

However, the model showed that delays alone had very limited predictive power.

This was an important moment in the analysis:
my initial assumption that delays dominate satisfaction turned out to be incorrect.

## Final logistic regression model

A richer model was then built using:

delays

PCA components

customer type

travel class

The final model achieved:

***Accuracy: 0.81***
***ROC-AUC: 0.88***

This indicates strong predictive capability.

## Key Findings

Several insights emerged from the model.

1.a. Travel class is the strongest driver of satisfaction

Passengers traveling in Economy and Economy Plus classes show significantly different satisfaction patterns.

Travel class had the largest coefficients in the model, indicating its strong influence.

## Customer type matters

Loyal customers behave differently from disloyal customers.

Customer type emerged as one of the strongest predictors of satisfaction.

This suggests that airlines benefit significantly from customer loyalty programs.

## Service quality matters more than delays

One of the most surprising findings was that:

service experience matters far more than delays

While delays had some effect, their coefficients were very small compared to:

service quality (PC1)

travel class

customer type

This suggests that passengers are more forgiving of delays if the overall service experience is strong.

## Hidden service dimensions explain satisfaction

PCA revealed that multiple service ratings actually represent broader underlying factors:

**overall onboard comfort**

**pre-boarding convenience**

These latent dimensions capture passenger experience more effectively than individual service metrics.

## Model Evaluation

The final model was evaluated using:

Confusion Matrix

Accuracy Score

ROC Curve

Feature coefficient analysis

The ROC curve achieved an AUC score of 0.88, indicating strong classification performance.

## Visualizations

The analysis includes:

PCA visualization of passenger satisfaction clusters

Logistic regression coefficient plots

Feature importance ranking

ROC curve evaluation

These visualizations help explain why the model behaves the way it does, not just how accurate it is.

## Key Learning Moments

During the project several analytical decisions changed based on evidence:

Initially assuming delays dominate satisfaction

Discovering instead that service quality is far more important

Using PCA to simplify correlated service ratings

Moving from a simple delay model to a richer behavioural model

These adjustments were important in improving both model performance and interpretability.

## Technologies Used

Python
Pandas
NumPy
Scikit-Learn
Matplotlib
Jupyter Notebook

## Future Improvements:

Possible extensions include:

Testing tree-based models (Random Forest, XGBoost)

Feature importance using SHAP values

Hyperparameter tuning

Investigating interaction effects between service factors

Author

Mehedi Hasan
Aspiring Data Scientist