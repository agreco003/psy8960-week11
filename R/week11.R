# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(caret)
library(haven)

# Data Import and Cleaning
gss_data <- read_spss("../data/GSS2016.sav") 
gss_tbl <- zap_missing(gss_data) %>% 
  rename(workhours = mosthrs) %>%
  drop_na(workhours) %>%
  select(workhours, which(colMeans(is.na(gss_data)) <= 0.25), -HRS1) %>%
  mutate(workhours = as.numeric(workhours))

#Visualization
histogram(gss_tbl$workhours, main = "Distribution of workhours")

# Machine Learning Models
set.seed(25)
index <- createDataPartition(gss_tbl$workhours, p = 0.75, list = FALSE)
train_tbl <- gss_tbl[index, ]
holdout_tbl <- gss_tbl[-index, ]
fold_indices = createFolds(train_tbl$workhours, k = 10)

## OLS Regression Model
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

## Elastic Net Model
en_model <- train(
  workhours ~ ., 
  data = train_tbl, 
  method = "glmnet",
  tuneLength = 3,
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

## Random Forest Model
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

## eXtreme Gradient Boosting Model
gb_model <- train(
  workhours ~ .,
  data = train_tbl,
  method = "xgbDART",
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

# Publication
model_list <- list(Linear = linear_model, ElasticNet = en_model, RandomForest = rf_model, GradientBoosting = gb_model)
results <- summary(resamples(model_list), metric="Rsquared")
dotplot(resamples(model_list), metric="Rsquared", main = "10-Fold CV Rsquared")
results

## Create tibble
cv_rsq <- results$statistics$Rsquared[,"Mean"] #mean values used because they correspond with each selected model, which minimizes RMSEA, as described by the each model output
ho_rsq <- c(r2_linear_holdout, r2_en_holdout, r2_rf_holdout, r2_gb_holdout)
table1_tbl <- tibble(algo = results$models, cv_rsq, ho_rsq) %>%
  mutate(cv_rsq = str_remove(format(round(cv_rsq, 2), nsmall = 2), "^0"),
         ho_rsq = str_remove(format(round(ho_rsq, 2), nsmall = 2), "^0"))
table1_tbl