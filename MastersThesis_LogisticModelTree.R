library(partykit)
library(dplyr)
library(caret)
library(pROC)
library(modelr)
library(purrr)
library(keras)

setwd("C:/Users/brian/Documents/TrainingData")
Training = read.csv("ProcessedData.csv")

#Indexing
Index = seq(1:68228)
Training = cbind(Index, Training)

CODs = Training[Training$Outcome.y == 1,] # splitting data due to imbalance
NoCOD = Training[Training$Outcome.y == 0,]

#Down Sampling larger class
under.balanced = NoCOD[sample(nrow(NoCOD), dim(CODs)[1]), ] # taking some of the rows from larger set 
Training.Undersampled = rbind.data.frame(CODs, under.balanced)
Training.Undersampled = Training.Undersampled[order(Training.Undersampled$Index),] # ordering by index so no order with underover
#Training.Undersampled = Training.Undersampled[-c(1)] #Removing index again
Training.Undersampled$Outcome.y = as.factor(Training.Undersampled$Outcome.y) #Making Outcome factor
levels(Training.Undersampled$Outcome.y) = c("noCOD", "yesCOD")
Training.Undersampled$Outcome.y <- relevel(Training.Undersampled$Outcome.y, ref = "yesCOD")

# Splitting into train and test sets
Train1 = Training.Undersampled %>% sample_frac(0.8) # splitting into train and test sets
Test1 = dplyr::anti_join(Training.Undersampled,Train1, by = 'Index')
#Remove Index col
Train1 = Train1[-c(1)]
Test1 = Test1[-c(1)]

LMTmodel <- glmtree(Outcome.y ~ .,
                    data = Train1, family = binomial(link = "logit"),
                    minsize = 50, maxdepth = 4, alpha = 0.05, prune = "AIC")

summary(LMTmodel)
coef(LMTmodel)
#Plot Decision Tree Model Schematic
plot(LMTmodel, type = "simple")

#Making Predictions on test data
pred <-predict(LMTmodel,newdata=Test1,type=c("response"))

#ROC curve
g <- roc(Outcome.y ~ pred, data = Test1)
plot(g, main = "LMT ROC Curve")

class_prediction <-
  ifelse(pred > 0.50,
         "yesCOD",
         "noCOD"
  )

class_prediction = as.factor(class_prediction)
class_prediction = relevel(class_prediction, "noCOD")
Test1$Outcome.y = relevel(Test1$Outcome.y, "noCOD")
levels(class_prediction)
levels(Test1$Outcome.y)

#Produce Confusion Matrix
confusionMatrix(data = class_prediction, reference = Test1$Outcome.y, positive = "noCOD")