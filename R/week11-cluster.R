# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(caret)
library(haven)
library(doParallel)

# Data Import and Cleaning
gss_data <- read_spss("../data/GSS2016.sav") 
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
  linear_model
  linear_predict <- predict(linear_model, holdout_tbl, na.action=na.pass)
  r2_linear_holdout <- cor(holdout_tbl$workhours, linear_predict)^2
  r2_linear_holdout
})
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
  en_model
  en_predict <- predict(en_model, holdout_tbl, na.action=na.pass)
  r2_en_holdout <- cor(holdout_tbl$workhours, en_predict)^2
  r2_en_holdout
})
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
  
  rf_model
  rf_predict <- predict(rf_model, holdout_tbl, na.action=na.pass)
  r2_rf_holdout <- cor(holdout_tbl$workhours, rf_predict)^2
  r2_rf_holdout
})
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
  gb_model
  gb_predict <- predict(gb_model, holdout_tbl, na.action=na.pass)
  r2_gb_holdout <- cor(holdout_tbl$workhours, gb_predict)^2
  r2_gb_holdout
})

## Parallelization
detectCores()
### OLS Regression Model
local_cluster <- makeCluster(10) #spreading across 10 cluster
registerDoParallel(local_cluster)
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
write.csv(table3_tbl, "table3.csv")

table4_tbl <- tibble(model = model_list, supercomputer = original, "supercomputer-10" = parallelized)
write.csv(table4_tbl, "table4.csv")