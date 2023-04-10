# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(caret)
library(haven)
library(doParallel)

# Data Import and Cleaning
gss_data <- read_spss("../data/GSS2016.sav") 
gss_tbl <- zap_missing(gss_data) %>% 
  rename(workhours =  MOSTHRS) %>% #changed to MOSTHRS variable per class demo
  drop_na(workhours) %>%
  select(workhours, which(colMeans(is.na(gss_data)) <= 0.25), -HRS1, -HRS2, -USUALHRS, -LEASTHRS, -SETHRS) %>% #removed all other references to work hours from code book, to be safe
  mutate(workhours = as.numeric(workhours))

# Visualization
histogram(gss_tbl$workhours, main = "Distribution of workhours") #added a title

# Machine Learning Models
set.seed(25)
index <- createDataPartition(gss_tbl$workhours, p = 0.75, list = FALSE)
train_tbl <- gss_tbl[index, ]
holdout_tbl <- gss_tbl[-index, ]
fold_indices = createFolds(train_tbl$workhours, k = 10)

## Original Core
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

## Parallelization
detectCores() #detect the number of cores on laptop
local_cluster <- makeCluster(7) #created 7 cores, #detectcores() - 1, just in case
registerDoParallel(local_cluster) #Dividing tasks across all 7 cores
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
})
linear_model
linear_predict <- predict(linear_model, holdout_tbl, na.action=na.pass)
r2_linear_holdout <- cor(holdout_tbl$workhours, linear_predict)^2
r2_linear_holdout
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
})
en_model
en_predict <- predict(en_model, holdout_tbl, na.action=na.pass)
r2_en_holdout <- cor(holdout_tbl$workhours, en_predict)^2
r2_en_holdout

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
})
rf_model
rf_predict <- predict(rf_model, holdout_tbl, na.action=na.pass)
r2_rf_holdout <- cor(holdout_tbl$workhours, rf_predict)^2
r2_rf_holdout

### eXtreme Gradient Boosting Model
gb_time_parallel <- system.time({
  gb_model <- train(
    workhours ~ .,
    data = train_tbl,
    method = "xgbTree",
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
### End Parallelization
stopCluster(local_cluster) #closes clusters
registerDoSEQ() #back to original setup

# Publication
## Variables
model_list <- list(Linear = linear_model, ElasticNet = en_model, RandomForest = rf_model, GradientBoosting = gb_model)
results <- summary(resamples(model_list), metric="Rsquared")
cv_rsq <- results$statistics$Rsquared[,"Mean"]
ho_rsq <- c(r2_linear_holdout, r2_en_holdout, r2_rf_holdout, r2_gb_holdout)
original <- c(ols_time_original[3], en_time_original[3], rf_time_original[3], gb_time_original[3]) # 3 = elapsed time within system.time() output
parallelized <-c(ols_time_parallel[3], en_time_parallel[3], rf_time_parallel[3], gb_time_parallel[3])
## Model Visualization
dotplot(resamples(model_list), metric="Rsquared", main = "10-Fold CV Rsquared")

## Tables
table1_tbl <- tibble(algo = results$models, cv_rsq, ho_rsq) %>%
  mutate(cv_rsq = str_remove(format(round(cv_rsq, 2), nsmall = 2), "^0"),
         ho_rsq = str_remove(format(round(ho_rsq, 2), nsmall = 2), "^0"))
table1_tbl

table2_tbl <- tibble('algo' = results$models, original = original, parallelized)
table2_tbl

## Questions
### 1. Which models benefited most from parallelization and why?
### The Elastic Net and Gradient Boosting models benefited the most from parallelization, cutting the analysis time in half and by about 2/3, respectively. For each of these models, different parameter configurations were able to evaluated concurrently on different cores, as each model was not dependent on the last. Notably, the random forest model using "ranger" did not benefit much. Perhaps, this was due to my own, idiosyncratic computer settings, which analyzed the random forest model across all 8 cores by default, or perhaps the default settings for the ranger package itself.

### 2. How big was the difference between the fastest and slowest parallelized model? Why?
### The slowest parallelized model was the gradient boosting model, requiring a few minutes, while the fastest was the elastic net model, requiring about 3 minutes. The elastic net model has fewer hyperparameters to test, and therefore fewer overall configurations to test. 

### 3. If your supervisor asked you to pick a model for use in a production model, which would you recommend and why? Consider both Table 1 and Table 2 when providing an answer.
### If I was asked to create a production model for this model specifically, I would first ask for additional context and check a few assumptions. Assuming that context isn't relevant, and assuming not everyone will use the same hardware as I am, I would recommend the parallelized random forest model. This model produces the best holdout sample R-squared value for this dataset and forces parallelization across multiple virtual cores, maximizing our model's prediction capability in the fastest way possible. Further, it exclusively uses local resources, which can be a benefit in some cases. I would also want to include a warning for users to be prompted once the analysis begins, informing them of the time this takes to complete.  