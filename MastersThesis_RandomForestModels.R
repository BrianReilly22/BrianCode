library(scales)
library(tidyverse)
library(modelr)
library(caret)
library(glmnet)
library(ranger)
library(randomForest)
library(ggplot2)
library(dplyr)
library(keras)
library(ROCR)

setwd("C:/Users/brian/Documents/TrainingData")
Training = read.csv("ProcessedData.csv")

#Indexing
Index = seq(1:68228)
Training = cbind(Index, Training)

CODs = Training[Training$Outcome.y == 1,] # splitting data due to imbalance
NoCOD = Training[Training$Outcome.y == 0,]

#Down Sampling Larger Class
under.balanced = NoCOD[sample(nrow(NoCOD), dim(CODs)[1]), ] # taking some of the rows from larger set 
Training.Undersampled = rbind.data.frame(CODs, under.balanced)
Training.Undersampled = Training.Undersampled[order(Training.Undersampled$Index),] # ordering by index so no order with underover
Training.Undersampled = Training.Undersampled[-c(1)] #Removing index again
Training.Undersampled$Outcome.y = as.factor(Training.Undersampled$Outcome.y) #Making Outcome factor
levels(Training.Undersampled$Outcome.y) = c("noCOD", "yesCOD")
Training.Undersampled$Outcome.y <- relevel(Training.Undersampled$Outcome.y, ref = "yesCOD")

## Random Forest Using Full Range of Features ##

#take off Adopted from variables names
variablesnames = colnames(Training.Undersampled)
variablesnames = variablesnames[-26]

# 5-fold 10-repeated CV
repeatedCV <- trainControl(method = "repeatedcv",
                           number = 5,
                           repeats = 10,
                           classProbs = TRUE,
                           summaryFunction=twoClassSummary)

#train the RF model
rf_new <- train(x = Training.Undersampled %>% select(all_of(variablesnames)),
                y = Training.Undersampled$Outcome.y,
                method = "rf",
                trControl = repeatedCV,
                importance = TRUE,
                metric = "ROC")


plot(rf_new)


#plot roc, sens, spec
plot(rf_new, metric = "Sens", main="Sensitivity")
plot(rf_new, metric = "Spec", main="Specificity")
plot(rf_new, metric = "ROC", main="ROC")

#confusion matrix
rf_new$finalModel$confusion

#error rates dataframe and plot
Final_model_error <- as.data.frame(cbind(ntree = 1:500, rf_new$finalModel$err.rate)) %>% 
  tidyr::gather(key = "Error_Type", value = "Error", -ntree)

Final_model_error$Error_Type[which(Final_model_error$Error_Type=='OOB')]='Out-Of-Bag'
Final_model_error$Error_Type[which(Final_model_error$Error_Type=='NO')] ='NotAdopted'
Final_model_error$Error_Type[which(Final_model_error$Error_Type=='YES')] ='Adopted'

ggplot(Final_model_error, aes(x = ntree, y = Error, col = factor(Error_Type))) +
  geom_line() +
  scale_y_continuous(labels = scales::percent, breaks = seq(0, 1, 0.05)) +
  labs(x = "Number of Trees", y = "Error Rate", colour = "Error Type") +
  ggtitle("RF Model (mtry=2) Error Rates")

rf_new$finalModel

varImp(rf_new)

## 2nd Random Forest: Using Smaller Set of Features ##
Small.Train.Under = Training.Undersampled[c(1,2,4,5,6,7,8,9,12,15,16,18,20,21,24,25,26)]
Small.Train.Under$Outcome.y = as.factor(Small.Train.Under$Outcome.y)
Small.Train.Under$Outcome.y <- relevel(Small.Train.Under$Outcome.y, ref = "yesCOD")

#take off Adopted from variables names
variablesnames1 = colnames(Small.Train.Under)
variablesnames1 = variablesnames1[-17]

#train the RF model
rf_new1 <- train(x = Small.Train.Under %>% select(all_of(variablesnames1)),
                 y = Small.Train.Under$Outcome.y,
                 method = "rf",
                 trControl = repeatedCV,
                 importance = TRUE,
                 metric = "ROC")


plot(rf_new1)


#plot roc, sens, spec
plot(rf_new1, metric = "Sens", main="Sensitivity")
plot(rf_new1, metric = "Spec", main="Specificity")
plot(rf_new1, metric = "ROC", main="ROC")

#confusion matrix
rf_new1$finalModel$confusion

#error rates dataframe and plot
Final_model_error1 <- as.data.frame(cbind(ntree = 1:500, rf_new1$finalModel$err.rate)) %>% 
  tidyr::gather(key = "Error_Type", value = "Error", -ntree)

Final_model_error1$Error_Type[which(Final_model_error1$Error_Type=='OOB')]='Out-Of-Bag'
Final_model_error1$Error_Type[which(Final_model_error1$Error_Type=='NO')] ='NotAdopted'
Final_model_error1$Error_Type[which(Final_model_error1$Error_Type=='YES')] ='Adopted'

ggplot(Final_model_error1, aes(x = ntree, y = Error, col = factor(Error_Type))) +
  geom_line() +
  scale_y_continuous(labels = scales::percent, breaks = seq(0, 1, 0.05)) +
  labs(x = "Number of Trees", y = "Error Rate", colour = "Error Type") +
  ggtitle("RF Model with Reduced Features (mtry=2) Error Rates")

rf_new1$finalModel

varImp(rf_new1)

#FullRF and ReducedFeaturesRF comparison#
# Compare both models performances using resamples()
models_compare <- resamples(list(All_Features=rf_new, Reduced_Features=rf_new1))
# Summary of the models performances and box plots to compare models
summary(models_compare)
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(models_compare, scales=scales, main = "Comparsion of Random Forest Models")

#Calculate 95% conf ints
n = sqrt(50)
trial = models_compare$values$`RandomForest2~ROC`
sd = sd(trial)
1.96*(sd/n)

## Random Forest Using only Accelerometer data ##
Accel.Only = Training.Undersampled[c(1:5,9:13,17:21,26)]

Accel.Only$Outcome.y = as.factor(Accel.Only$Outcome.y)
Accel.Only$Outcome.y <- relevel(Accel.Only$Outcome.y, ref = "yesCOD")

#take off Adopted from variables names
variablesnamesA = colnames(Accel.Only)
variablesnamesA = variablesnamesA[-16]

#train the RF model
rf_newA <- train(x = Accel.Only %>% select(all_of(variablesnamesA)),
                 y = Accel.Only$Outcome.y,
                 method = "rf",
                 trControl = repeatedCV,
                 importance = TRUE,
                 metric = "ROC")


plot(rf_newA)


#plot roc, sens, spec
plot(rf_newA, metric = "Sens", main="Sensitivity")
plot(rf_newA, metric = "Spec", main="Specificity")
plot(rf_newA, metric = "ROC", main="ROC")

#confusion matrix
rf_newA$finalModel$confusion

#error rates dataframe and plot
Final_model_errorA <- as.data.frame(cbind(ntree = 1:500, rf_newA$finalModel$err.rate)) %>% 
  tidyr::gather(key = "Error_Type", value = "Error", -ntree)

Final_model_errorA$Error_Type[which(Final_model_errorA$Error_Type=='OOB')]='Out-Of-Bag'
Final_model_errorA$Error_Type[which(Final_model_errorA$Error_Type=='NO')] ='NotAdopted'
Final_model_errorA$Error_Type[which(Final_model_errorA$Error_Type=='YES')] ='Adopted'

ggplot(Final_model_errorA, aes(x = ntree, y = Error, col = factor(Error_Type))) +
  geom_line() +
  scale_y_continuous(labels = scales::percent, breaks = seq(0, 1, 0.05)) +
  labs(x = "Number of Trees", y = "Error Rate", colour = "Error Type") +
  ggtitle("RF Model with Accelerometer Features (mtry=2) Error Rates")

rf_newA$finalModel

varImp(rf_newA)

## Random Forest Using only Gyro Data
Gyro.Only = Training.Undersampled[c(6:8,14:16,22:24,26)]

Gyro.Only$Outcome.y = as.factor(Gyro.Only$Outcome.y)
Gyro.Only$Outcome.y <- relevel(Gyro.Only$Outcome.y, ref = "yesCOD")

#take off Adopted from variables names
variablesnamesG = colnames(Gyro.Only)
variablesnamesG = variablesnamesG[-10]

#train the RF model
rf_newG <- train(x = Gyro.Only %>% select(all_of(variablesnamesG)),
                 y = Gyro.Only$Outcome.y,
                 method = "rf",
                 trControl = repeatedCV,
                 importance = TRUE,
                 metric = "ROC")


plot(rf_newG)


#plot roc, sens, spec
plot(rf_newG, metric = "Sens", main="Sensitivity")
plot(rf_newG, metric = "Spec", main="Specificity")
plot(rf_newG, metric = "ROC", main="ROC")

#confusion matrix
rf_newG$finalModel$confusion

#error rates dataframe and plot
Final_model_errorG <- as.data.frame(cbind(ntree = 1:500, rf_newG$finalModel$err.rate)) %>% 
  tidyr::gather(key = "Error_Type", value = "Error", -ntree)

Final_model_errorG$Error_Type[which(Final_model_errorG$Error_Type=='OOB')]='Out-Of-Bag'
Final_model_errorG$Error_Type[which(Final_model_errorG$Error_Type=='NO')] ='NotAdopted'
Final_model_errorG$Error_Type[which(Final_model_errorG$Error_Type=='YES')] ='Adopted'

ggplot(Final_model_errorG, aes(x = ntree, y = Error, col = factor(Error_Type))) +
  geom_line() +
  scale_y_continuous(labels = scales::percent, breaks = seq(0, 1, 0.05)) +
  labs(x = "Number of Trees", y = "Error Rate", colour = "Error Type") +
  ggtitle("RF Model with Gyroscopic Features (mtry=2) Error Rates")

rf_newG$finalModel

varImp(rf_newG)

#Accelerometer and Gyroscopic Models Comparison #
# Compare both models performances using resamples()
models_compare1 <- resamples(list(Accel_Vars=rf_newA, Gyro_Vars=rf_newG))
# Summary of the models performances and box plots to compare models
summary(models_compare1)
bwplot(models_compare1, scales=scales, main = "Comparsion of Random Forest Models")
