# Yellow New York Taxi Demand multiple Forecasting with spatio-temporal Transformers

Having been involved in a couple of project for multiple time series forecasting I was always wondering how possible cross relationship between being predicted values can be taken into account. It is rather difficult in case of even moderate amount of predicted values it could be done only by appealing to domain knowledge. Attention mechanisms (Transformers) seem to fit this problem well, and there are also many articles dealing with this topic.

This personal project had two objectives: 
- to attempt to apply transformers to a multiple time series forecasting problem
- to practise developing end-to-end ML projects with a focus on development aspects and MLOps

As a game problem, I took forecasting the demand for yellow NYC taxis. I was familiar with the dataset as I had developed a forecasting algorithm in 2020 (it was my first ML project), which I could also use now as a base-line solution.