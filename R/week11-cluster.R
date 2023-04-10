# Script Settings and Resources
library(tidyverse)
library(caret)
library(haven)
library(doParallel)

# Data Import and Cleaning
gss_data <- read_spss("GSS2016.sav") 
gss_tbl <- zap_missing(gss_data) %>% 
  rename(workhours =  MOSTHRS) %>%
  drop_na(workhours) %>%
  select(workhours, which(colMeans(is.na(gss_data)) <= 0.25), -HRS1, -HRS2, -USUALHRS, -LEASTHRS, -SETHRS) %>%
  mutate(workhours = as.numeric(workhours))

# Machine Learning Models
set.seed(25)
index <- createDataPartition(gss_tbl$workhours, p = 0.75, list = FALSE)
train_tbl <- gss_tbl[index, ]
holdout_tbl <- gss_tbl[-index, ]
fold_indices = createFolds(train_tbl$workhours, k = 10)

## Original Single Core
### OLS Regression Model
ols_time_original <- system.time({
  linear_model <- train(
    workhours ~ ., 
    data = train_tbl, 
    method = "lm",
    na.action = "na.pass", 
    preProcess = "medianImpute",
    trControl = trainControl(
      method = "cv", 
      indexOut = fold_indices,
      verboseIter = TRUE
    )
  )
})
  linear_model
  linear_predict <- predict(linear_model, holdout_tbl, na.action=na.pass)
  r2_linear_holdout <- cor(holdout_tbl$workhours, linear_predict)^2
  r2_linear_holdout
### Elastic Net Model
en_time_original <- system.time({
  en_model <- train(
    workhours ~ ., 
    data = train_tbl, 
    method = "glmnet",
    tuneLength = 4,
    na.action = "na.pass", 
    preProcess = "medianImpute",
    trControl =  trainControl(
      method = "cv",
      indexOut = fold_indices,
      verboseIter = TRUE
    )
  )
})
  en_model
  en_predict <- predict(en_model, holdout_tbl, na.action=na.pass)
  r2_en_holdout <- cor(holdout_tbl$workhours, en_predict)^2
  r2_en_holdout
  
### Random Forest Model
rf_time_original <- system.time({
  rf_model <- train(
    workhours ~ ., 
    data = train_tbl, 
    tuneLength = 5,
    na.action = "na.pass", 
    preProcess = "medianImpute", 
    method = "ranger",
    trControl = trainControl(
      method = "cv", 
      indexOut = fold_indices,
      verboseIter = TRUE
    )
  )
})
  rf_model
  rf_predict <- predict(rf_model, holdout_tbl, na.action=na.pass)
  r2_rf_holdout <- cor(holdout_tbl$workhours, rf_predict)^2
  r2_rf_holdout
  
### eXtreme Gradient Boosting Model
gb_time_original <- system.time({
  gb_model <- train(
    workhours ~ .,
    data = train_tbl,
    method = "xgbLinear",
    tuneLength = 5,
    na.action = "na.pass",
    preProcess = "medianImpute",
    trControl =  trainControl(
      method = "cv",
      indexOut = fold_indices,
      verboseIter = TRUE
    )
  )
})
  gb_model
  gb_predict <- predict(gb_model, holdout_tbl, na.action=na.pass)
  r2_gb_holdout <- cor(holdout_tbl$workhours, gb_predict)^2
  r2_gb_holdout

## Virtual Cores
detectCores()
local_cluster <- makeCluster(12) #spreading across 12 virtual cores! Could use more if desired. 
registerDoParallel(local_cluster)
### OLS Regression Model
ols_time_parallel <- system.time({
  linear_model <- train(
    workhours ~ ., 
    data = train_tbl, 
    method = "lm",
    na.action = "na.pass", 
    preProcess = "medianImpute",
    trControl = trainControl(
      method = "cv", 
      indexOut = fold_indices,
      verboseIter = TRUE
    )
  )
  linear_model
  linear_predict <- predict(linear_model, holdout_tbl, na.action=na.pass)
  r2_linear_holdout <- cor(holdout_tbl$workhours, linear_predict)^2
  r2_linear_holdout
})
### Elastic Net Model
en_time_parallel <- system.time({
  en_model <- train(
    workhours ~ ., 
    data = train_tbl, 
    method = "glmnet",
    tuneLength = 5,
    na.action = "na.pass", 
    preProcess = "medianImpute",
    trControl =  trainControl(
      method = "cv",
      indexOut = fold_indices,
      verboseIter = TRUE
    )
  )
  en_model
  en_predict <- predict(en_model, holdout_tbl, na.action=na.pass)
  r2_en_holdout <- cor(holdout_tbl$workhours, en_predict)^2
  r2_en_holdout
})
### Random Forest Model
rf_time_parallel <- system.time({
  rf_model <- train(
    workhours ~ ., 
    data = train_tbl, 
    tuneLength = 5,
    na.action = "na.pass", 
    preProcess = "medianImpute", 
    method = "ranger",
    trControl = trainControl(
      method = "cv", 
      indexOut = fold_indices,
      verboseIter = TRUE
    )
  )
  
  rf_model
  rf_predict <- predict(rf_model, holdout_tbl, na.action=na.pass)
  r2_rf_holdout <- cor(holdout_tbl$workhours, rf_predict)^2
  r2_rf_holdout
})
### eXtreme Gradient Boosting Model
gb_time_parallel <- system.time({
  gb_model <- train(
    workhours ~ .,
    data = train_tbl,
    method = "xgbLinear",
    tuneLength = 5,
    na.action = "na.pass",
    preProcess = "medianImpute",
    trControl =  trainControl(
      method = "cv",
      indexOut = fold_indices,
      verboseIter = TRUE
    )
  )
  gb_model
  gb_predict <- predict(gb_model, holdout_tbl, na.action=na.pass)
  r2_gb_holdout <- cor(holdout_tbl$workhours, gb_predict)^2
  r2_gb_holdout
})
### End Virtual Cores
stopCluster(local_cluster)
registerDoSEQ()

# Publication
## Variables
model_list <- list(Linear = linear_model, ElasticNet = en_model, RandomForest = rf_model, GradientBoosting = gb_model)
results <- summary(resamples(model_list), metric="Rsquared")
results

cv_rsq <- results$statistics$Rsquared[,"Mean"]
ho_rsq <- c(r2_linear_holdout, r2_en_holdout, r2_rf_holdout, r2_gb_holdout)
original <- c(ols_time_original[3], en_time_original[3], rf_time_original[3], gb_time_original[3])
parallelized <-c(ols_time_parallel[3], en_time_parallel[3], rf_time_parallel[3], gb_time_parallel[3])

## Tables
table3_tbl <- tibble(algo = results$models, cv_rsq, ho_rsq) %>%
  mutate(cv_rsq = str_remove(format(round(cv_rsq, 2), nsmall = 2), "^0"),
         ho_rsq = str_remove(format(round(ho_rsq, 2), nsmall = 2), "^0"))
write_csv(table3_tbl, "table3.csv")

table4_tbl <- tibble(algo = results$models, supercomputer = original, 'supercomputer-12' = parallelized)
write_csv(table4_tbl, "table4.csv")

## Questions
### 1. Which models benefited most from moving to the supercomputer and why?
### The Elastic Net, Random Forest, and Gradient Boosting models all benefited by moving to the supercomputer. In terms of the raw time required for each analysis, the Gradient Boosting model benefited the most. However, all of these models benefited from the additional resources, including the additional cores, made available by using the supercomputer. Complex tasks and configurations were able to be parsed and run simultaneously across additional virtual cores. This resulted in a faster time to completion, more so than even the parallelized version in the previous analysis. 

### 2. What is the relationship between time and the number of cores used?
### Generally speaking, the more cores used, the less time an analysis will take. However, there is a communication cost between cores to consider, particularly when simple models or strictly-sequential tasks are needed. In these instances, when a single core quickly completes a task on its own, coordinating these additional cores will actually increase the time to completion, as was the case for the original lm model. 

### 3. If your supervisor asked you to pick a model for use in a production model, would you recommend using the supercomputer and why? Consider all four tables when providing an answer.
### Again, not withstanding any changes in context or assumptions, like the user population having VPN access to the supercomputer and stable internet connection or the number of other jobs on the supercomputer, I would recommend that we run the random forest model on our supercomputer, using the 12, virtual core configuration. This still maximizes our prediction capability, as measured by R-squared, and also reduces the run time by a significant amount. Further, if desired and the resources were available, we could maximize this efficiency by increasing the number of cores used and their memory to make these predictions even faster, again assuming we had access to these additional resources in our company. 