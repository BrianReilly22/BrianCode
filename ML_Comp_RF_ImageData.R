library(ranger)
library(tidyverse)

#Image Features Data
features = read.csv("200331_features_pretrained_cnn_trunc.csv")
testfeatures = features[38102:57571,]
trainfeatures = features[1:38101,]
newtestbind = testfeatures[,-2]
newtrainbind = trainfeatures[,-2]

#Image features Data
newtestfinal = aggregate(newtestbind, by=list(newtestbind$PetID), FUN = mean)
newtestfinal = newtestfinal[,-2]
names(newtestfinal)[1] = "PetID"
newtrainfinal = aggregate(newtrainbind, by=list(newtrainbind$PetID), FUN = mean)
newtrainfinal = newtrainfinal[,-2]
names(newtrainfinal)[1] = "PetID"

#Reading in Tabular Data
Train = read.csv("NewTrainSet")
Adopted = Train[,c(1,20)]
Test = read.csv("NewTestSet.csv")
FeautresTrain = merge(newtrainfinal, Adopted, by = "PetID")
FeautresTest = newtestfinal
FeautresTrain$Adopted = as.integer(as.factor(FeautresTrain$Adopted))
FeautresTrain$Adopted = (FeautresTrain$Adopted - 1)

#Create Random Forest model using ranger
mdl_rang = ranger(Adopted ~ ., data = FeautresTrain, probability = T)

#Training Predictions
TrainPreds = predict(mdl_rang, FeautresTrain, checkpoint_path = NULL,
                     predict_keys = c("response"), hooks = NULL, as_iterable = FALSE,
                     simplify = TRUE, yield_single_examples = TRUE)
TrainingPredictions = TrainPreds$predictions
TrainingPredictions = TrainingPredictions[,-2]
Train1 = Train[Train$PhotoAmt != 0,]
Train1 = cbind(Train1, TrainingPredictions)
names(Train1)[24] = "NewImage"
Train1 = Train1[,-23]

#Adding rows with 0 photos back to training data
ZeroPics = Train[Train$PhotoAmt == 0,]
NewTrain = dplyr::bind_rows(Train1, ZeroPics)
NewTrain = NewTrain %>% arrange(PetID)

#Test Predictions
TestPreds = predict(mdl_rang, FeautresTest, checkpoint_path = NULL,
                    predict_keys = c("response"), hooks = NULL, as_iterable = FALSE,
                    simplify = TRUE, yield_single_examples = TRUE)
TestPredictions = TestPreds$predictions
TestPredictions = TestPredictions[,-2]
Test1 = Test[Test$PhotoAmt != 0,]
Test1 = cbind(Test1, TestPredictions)
names(Test1)[23] = "NewImage"
Test1 = Test1[,-22]

#Adding rows with 0 photos back to training data
ZeroPics1 = Test[Test$PhotoAmt == 0,]
NewTest = dplyr::bind_rows(Test1, ZeroPics1)
NewTest = NewTest %>% arrange(PetID)

#Saving Files
write.csv(x=NewTrain, file = "C:/Users/brian/Downloads/TrainingData.csv", row.names =
            FALSE)
write.csv(x=NewTest, file = "C:/Users/brian/Downloads/TestingData.csv", row.names =
            FALSE)