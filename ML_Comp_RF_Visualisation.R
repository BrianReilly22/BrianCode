library(scales)
library(tidyverse)
library(mlr)
library(randomForest)
library(caret)

#PREPROCESSING TRAIN DATA (sentiment data included - NewTrain)
train <- read.csv("NewTrain.csv" , stringsAsFactors = FALSE)

#Remove Color2
train$Color2 = NULL

#factors
train$Type = as.factor(train$Type)
train$Gender = as.factor(train$Gender)
train$Color1 = as.factor(train$Color1)
train$Color3 = as.factor(train$Color3)
train$MaturitySize = as.factor(train$MaturitySize)
train$FurLength = as.factor(train$FurLength)
train$Vaccinated = as.factor(train$Vaccinated)
train$Dewormed = as.factor(train$Dewormed)
train$Sterilized = as.factor(train$Sterilized)
train$State = as.factor(train$State)
train$Adopted = as.factor(train$Adopted)
train$Health = as.factor(train$Health)
train = na.omit(train) #9405 obs

# bind with Image Prediction Vector from CNN to get new train set
trainfeatures = read.csv('TrainImageFeaturesIvan.csv')
trainfeatures = trainfeatures %>% rename(PetID = Group.1)
trainfeatures$PetID = as.character(trainfeatures$PetID)
newtrain = inner_join(trainfeatures, train, by = "PetID")

#petID no need for model
newtrain$PetID = NULL #9204 obs
set.seed(2307)

# 5-fold 5-repeated CV
repeatedCV <- trainControl(method = "repeatedcv",
                           number = 5,
                           repeats = 5,
                           classProbs = TRUE,
                           summaryFunction=twoClassSummary)

# select mtry
rf_new_grid <- expand.grid(mtry = seq(from = 7, to = 42, by = 5))

#take off Adopted from variables names
variablesnames = colnames(newtrain)
variablesnames = variablesnames[-274]

#train the model
rf_new <- train(x = newtrain %>% select(variablesnames),
                y = newtrain$Adopted,
                method = "rf",
                trControl = repeatedCV,
                importance = TRUE,
                metric = "ROC",
                tuneGrid = rf_new_grid)

#plot the relative variable importances
plot(varImp(rf_new), top = 20)

#plot metrics over different mtry
plot(rf_new )

#confusion matrix
rf_new$finalModel$confusion

#error rates – prepare data frame
Final_model_error <- as.data.frame(cbind(ntree = 1:500, rf_new$finalModel$err.rate)) %>%
  tidyr::gather(key = "Error_Type", value = "Error", -ntree)
Final_model_error$Error_Type[which(Final_model_error$Error_Type == 'OOB')] <-'Out-Of-Bag'
Final_model_error$Error_Type[which(Final_model_error$Error_Type == 'NO')] <-'NotAdopted'
Final_model_error$Error_Type[which(Final_model_error$Error_Type == 'YES')] <-'Adopted'

#error rates – plot
ggplot(Final_model_error, aes(x = ntree, y = Error, col = factor(Error_Type))) +
  geom_line() +
  scale_y_continuous(labels = scales::percent, breaks = seq(0, 1, 0.05)) +
  labs(x = "Number of Trees", y = "Error Rate", colour = "Error Type") +
  ggtitle("RF Model1 (mtry=37) Error Rates")

#plot roc, sens, spec
plot(rf_new, metric = "Sens", main="Sensitivity")
plot(rf_new, metric = "Spec", main="Specificity")
plot(rf_new, metric = "ROC", main="ROC")
# RF MODEL 4

#PREPROCESSING TRAIN DATA (TrainingData contains extra column with image predictions)
trainBR <- read.csv("TrainingData.csv" , stringsAsFactors = FALSE)
trainBR$Color2 = NULL

#factors
trainBR$Type = as.factor(trainBR$Type)
trainBR$Gender = as.factor(trainBR$Gender)
trainBR$Color1 = as.factor(trainBR$Color1)
trainBR$Color3 = as.factor(trainBR$Color3)
trainBR$MaturitySize = as.factor(trainBR$MaturitySize)
trainBR$FurLength = as.factor(trainBR$FurLength)
trainBR$Vaccinated = as.factor(trainBR$Vaccinated)
trainBR$Dewormed = as.factor(trainBR$Dewormed)
trainBR$Sterilized = as.factor(trainBR$Sterilized)
trainBR$State = as.factor(trainBR$State)
trainBR$Adopted = as.factor(trainBR$Adopted)
trainBR$Health = as.factor(trainBR$Health)
trainBR = na.omit(trainBR)
trainBR$PetID = NULL

#Set mtry values
rf_new_gridBR <- expand.grid(mtry = c(2, 3, 4, 5, 6, 7, 8))

#take off Adopted from variables names
variablesnamesBR = colnames(trainBR)
variablesnamesBR = variablesnamesBR[-18]

#train the RF model 4
rf_newBR <- train(x = trainBR %>% select(variablesnamesBR),
                  y = trainBR$Adopted,
                  method = "rf",
                  trControl = repeatedCV,
                  importance = TRUE,
                  metric = "ROC",
                  tuneGrid = rf_new_gridBR)

#plot the relative variable importances
plot(varImp(rf_newBR))

#plot roc, sens, spec
plot(rf_newBR, metric = "Sens", main="Sensitivity")
plot(rf_newBR, metric = "Spec", main="Specificity")
plot(rf_newBR, metric = "ROC", main="ROC")

#confusion matrix
rf_newBR$finalModel$confusion

#error rates dataframe and plot
Final_model_errorBR <- as.data.frame(cbind(ntree = 1:500, rf_newBR$finalModel$err.rate))
%>% tidyr::gather(key = "Error_Type", value = "Error", -ntree)
Final_model_errorBR$Error_Type[which(Final_model_errorBR$Error_Type=='OOB')]='Out-Of-Bag'
Final_model_errorBR$Error_Type[which(Final_model_errorBR$Error_Type=='NO')] ='NotAdopted'
Final_model_errorBR$Error_Type[which(Final_model_errorBR$Error_Type=='YES')] ='Adopted'
ggplot(Final_model_errorBR, aes(x = ntree, y = Error, col = factor(Error_Type))) +
  geom_line() +
  scale_y_continuous(labels = scales::percent, breaks = seq(0, 1, 0.05)) +
  labs(x = "Number of Trees", y = "Error Rate", colour = "Error Type") +
  ggtitle("RF Model 2 (mtry=6) Error Rates")

# Compare both models performances using resamples()
models_compare <- resamples(list(RandomForest1=rf_new, RandomForest2=rf_newBR))

# Summary of the models performances and box plots to compare models
summary(models_compare)
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(models_compare, scales=scales)