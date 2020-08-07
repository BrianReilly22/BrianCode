library(scales)
library(tidyverse)
library(modelr)
library(caret)
library(glmnet)

# Read in original train data
train = read.csv('train.csv')

# PREPROCESSING - Replacing text with numeric
train$Adopted = as.integer(as.factor(train$Adopted))

# Converting to categorical data
train$Type = as.factor(train$Type)
train$Breed1 = as.factor(train$Breed1)
train$Breed2 = as.factor(train$Breed2)
train$Gender = as.factor(train$Gender)
train$Color1 = as.factor(train$Color1)
train$Color2 = as.factor(train$Color2)
train$Color3 = as.factor(train$Color3)
train$MaturitySize = as.factor(train$MaturitySize)
train$FurLength = as.factor(train$FurLength)
train$Vaccinated = as.factor(train$Vaccinated)
train$Dewormed = as.factor(train$Dewormed)
train$Sterilized = as.factor(train$Sterilized)
train$Health = as.factor(train$Health)
train$State = as.factor(train$State)
train$PetID = as.factor(train$PetID)
train$Adopted = as.factor(train$Adopted)

# Dealing with missing values
train$Breed1[train$Breed1 == 0] = NA
train$Breed1[train$Breed1 == 307] = NA
train$Breed2[train$Breed2 == 0] = NA
train$Breed2[train$Breed2 == 307] = NA
train$Color2[train$Color2 == 0] = NA
train$Color3[train$Color3 == 0] = NA
train1 = na.omit(train) #only 539 variables left
train1$PetID = NULL
train1$Color3 = NULL

# Run GLM model
modelglm = glm(Adopted ~ . , data = train1, family = binomial(link='logit'))

# Stepwise regression used to choose important predictive variables
step = step(modelglm)

# Variable importance
varImp(step)

# Set seed for reproducibility
set.seed(123)

# Split train1 in 10 test data sets
cv_k10 = crossv_kfold(train1, k = 10)

# Train models (all variables and stepwise)
model1_logreg = map(cv_k10$train, ~ glm(Adopted ~ . , data = ., family = binomial(link='logit')))
model2_step = map(cv_k10$train, ~ glm(Adopted ~ FurLength + Sterilized + Fee, data = ., f
                                      amily = binomial(link='logit')))
# Function for data preparation
model_glmnet = function(data, alpha){
  data = as.data.frame(data)
  X_mdl_k = model.matrix(Adopted ~ . , data = data)
  y_mdl_k = factor(data$Adopted)
  mdl_k = cv.glmnet(X_mdl_k, y_mdl_k, alpha = alpha, family = "binomial")
}

# Train models (ridge penalisation and lasso penalisation)
model3_ridge = map(cv_k10$train, model_glmnet, alpha = 0)
model4_lasso = map(cv_k10$train, model_glmnet, alpha = 1)

# AUC performance metric
auc_glm = function(model, data){
  library(ROCR)
  data = as.data.frame(data)
  probs = predict(model, newdata = data, type = "response")
  preds = prediction(predictions = probs, labels = data$Adopted)
  t = performance(preds, "auc")
  t@y.values[[1]]
}

# AUC for logreg and stepwise models
auc_model1 = map2_dbl(model1_logreg, cv_k10$test, auc_glm)
auc_model2 = map2_dbl(model2_step, cv_k10$test, auc_glm)
auc_glmnet = function(model, data){
  library(ROCR)
  data = as.data.frame(data)
  X_tst_k = model.matrix(Adopted ~ . , data = data)
  probs = predict(model, newx = X_tst_k, type = "response")
  preds = prediction(predictions = probs, labels = data$Adopted)
  t = performance(preds, "auc")
  t@y.values[[1]]
}

# AUC for ridge and lasso models
auc_model3 = map2_dbl(model3_ridge, cv_k10$test, auc_glmnet)
auc_model4 = map2_dbl(model4_lasso, cv_k10$test, auc_glmnet)

# AUC boxplot comparison
perfc_df = data.frame(iter = 1:10, model1 = auc_model1, model2 = auc_model2, model3 = auc
                      _model3, model4 = auc_model4)
perfc_df = perfc_df %>% pivot_longer(cols = model1:model4, names_to = "Model", values_to
                                     = "AUC")
perfc_df %>%
  ggplot(aes(x = Model, y = AUC, fill = Model)) + geom_boxplot() + scale_fill_manual(values=c("lightpink1", "lightskyblue", "wheat1", "palegreen1"), name="Logistic Regression",
                                                                                     breaks=c("model1", "model2","model3","model4"),
                                                                                     labels=c("Full range of variables", "Stepwise variables", "Ridge penalisation", "Lasso penalisation"))


# PREDICTIONS - Preprocessing test data
test = read.csv('test.csv')

#Converting to categorical data
test$Type = as.factor(test$Type)
test$Breed1 = as.factor(test$Breed1)
test$Breed2 = as.factor(test$Breed2)
test$Gender = as.factor(test$Gender)
test$Color1 = as.factor(test$Color1)
test$Color2 = as.factor(test$Color2)
test$Color3 = as.factor(test$Color3)
test$MaturitySize = as.factor(test$MaturitySize)
test$FurLength = as.factor(test$FurLength)
test$Vaccinated = as.factor(test$Vaccinated)
test$Dewormed = as.factor(test$Dewormed)
test$Sterilized = as.factor(test$Sterilized)
test$Health = as.factor(test$Health)
test$State = as.factor(test$State)
test$PetID = as.factor(test$PetID)
#Dealing with missing values
test$Breed1[test$Breed1 == 0] = NA
test$Breed1[test$Breed1 == 307] = NA
test$Breed2[test$Breed2 == 0] = NA
test$Breed2[test$Breed2 == 307] = NA
test$Color2[test$Color2 == 0] = NA
test$Color3[test$Color3 == 0] = NA

#Prediciting adoption probabilities for Test data
predictions = predict(model2_step, newdata = test, type = "response")