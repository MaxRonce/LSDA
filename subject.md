Large-Scale Data Analysis
Project 2025-2026
1 Introduction
This project addresses a large-scale data classification problem: distinguishing between
Higgs boson signal processes and background processes. The primary objective is to
evaluate the trade-offs between traditional SciKit-Learn and distributed PySpark pipelines.
To ensure a rigorous comparison, we will implement three consistent classification methods
across both frameworks, resulting in six tuned models. Our analysis will focus on two key
dimensions: predictive performance (Accuracy and ROC scores) and computational
efficiency (training and inference time). Through this comparison, we aim to gain insights
into the most effective learning strategies for large-scale datasets.
2 Dataset
The Higgs Boson Dataset [1] has been produced using Monte Carlo simulations. The first
21 features (columns 2-22) are kinematic properties measured by the particle detectors in the
accelerator. The last 7 features are functions of the first 21 features; these are high-level
features derived by physicists to help discriminate between the two classes. There is an
interest in using machine learning methods to obviate the need for physicists to manually
develop such features.
The standard benchmark results using Bayesian Decision Trees from a standard physics
package and 5-layer neural networks are presented in the original paper [2]. The full dataset
contains 11,000,000 examples and the last 500,000 examples are used as a test set. In this
project, we use an extract of 2,000,000 examples of the full dataset as the training set and
500,000 examples as the test set. These two datasets are available at:
https://utbox.univ-tours.fr/s/dXYJaT2x7fkaxGm
3 Project Tasks
1. Dataset Analysis
a. Perform a comprehensive analysis of the dataset based on the provided
physical descriptions.
b. Conduct statistical analysis to understand the distribution of signal vs.
background processes and identify potential challenges in the raw data.
2. Comparative Feature Engineering Pipelines
a. Construct two distinct feature engineering pipelines—one using SciKit-Learn
and one using PySpark ML.
Université de Tours Master Data Sciences For Societal Challenges
2
b. Both pipelines must implement identical transformations, including feature
vectorization (e.g., VectorAssembler in Spark) and feature scaling (e.g.,
StandardScaler).
c. Explain the selection of features and justify why certain attributes are critical
for distinguishing Higgs boson production.
3. Model Development, Tuning, and Cross-Validation
a. Implement three consistent classification methods (e.g., Logistic Regression,
Random Forest, and Gradient Boosted Trees, but not limited to) across both
frameworks, resulting in a total of six models.
b. For each method, perform K-fold cross-validation and parameter tuning to
identify the optimal model configuration.
c. Ensure that the search space for hyperparameters is consistent between the
SciKit-Learn and PySpark implementations to allow for a fair comparison.
4. Computational Efficiency and Scalability Benchmarking
a. Conduct a benchmarking study by measuring and reporting the training and
prediction times relative to the number of CPU cores utilized.
b. For SciKit-Learn, control parallelism using the n_jobs parameter.
c. For PySpark, configure the SparkSession using different local[k] master
configurations (e.g., local[1] vs. local[2]).
d. Discuss the impact of parallel speedup when handling 2,000,000 training
records.
5. Performance Evaluation and Framework Comparison
a. Apply the six tuned models to the test dataset (500,000 records) and compare
their performance using Accuracy and ROC-AUC scores.
b. Provide a critical analysis of the results, focusing on:
i. The consistency of predictive accuracy between local and distributed
frameworks.
ii. The time-efficiency trade-offs of using PySpark for large-scale learning
vs. SciKit-Learn in a resource-constrained environment like Google
Colab.