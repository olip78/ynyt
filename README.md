# Yellow New York Taxi Demand multiple Forecasting with spatio-temporal Transformers

**Dashboard:** [link](http://158.160.113.119:8080)

**Tech stack:** Sklearn, Pytorch, MLflow, Docker, Swarm, Gitlab CI/CD
Having participated in several projects involving the forecasting of multiple time series, I've consistently pondered the challenge of incorporating cross-relationships among predicted values. This task proves intricate, and even in case of moderate amount of predicted values it could be done only by appealing to domain knowledge. The potential of transformers in addressing this issue has caught my attention, supported by numerous articles exploring their efficacy in such scenarios.

In pursuit of these considerations, my personal project sought to achieve two primary objectives:

1. Experiment with the application of transformers to address the complexities of multiple time series forecasting.
2. Hone my skills in developing end-to-end machine learning projects, emphasizing both development aspects and the implementation of MLOps practices.

As a game problem, I took forecasting the demand for yellow NYC taxis. I was familiar with the dataset as I had developed a forecasting algorithm in 2020 (it was my first ML project), which I could also use now as a baseline solution. 

## Table of Contents
- [Yellow New York Taxi Demand multiple Forecasting with spatio-temporal Transformers](#yellow-new-york-taxi-demand-multiple-forecasting-with-spatio-temporal-transformers)
  - [Table of Contents](#table-of-contents)
    - [Problem statement ](#problem-statement-)
    - [Baseline Solution ](#baseline-solution-)
    - [Spatio-Temporal Transformers ](#spatio-temporal-transformers-)
    - [App Schema and Deployment ](#app-schema-and-deployment-)

### Problem statement <a name="problem_statement"></a>

Disclaimer: Since the goal was to experiment with different types of approaches and I was interested in relative quality, I did not attempt to use any external information (e.g., weather forecasts) and limited myself to data from the original dataset.

### Baseline Solution <a name="baseline"></a>
Experimenting with different models, I settled on a simple ridge regression with complex features. The solution had two key idea:
1. using Fourier harmonics to model large periods
2. using the Cartesian product for the baseline features and zones id / hours
   
A base features I took autoregressive number of drives (-1, -2, ..., -24, -48, ... hours) as well as average journeys information for the last hour (distance, cost, drive time, etc.) 

### Spatio-Temporal Transformers <a name="transformers"></a>


### App Schema and Deployment <a name="deployment"></a>