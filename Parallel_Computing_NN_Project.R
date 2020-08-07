# Please note this script is not meant to be run and is a demonstration of the code used across the different models #

'### DATA CLEANING AND ENCODING ###'

## Installing data ##

adultcensus = as.data.frame(read.table('adult.data', sep = ',', na.strings=c(" ?"))) # read in and get rid of some NA's
columnnames = c('age', 'workclass', 'fnlwgt', 'education','educational-numeric ',  'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native country','underover')
colnames(adultcensus) = columnnames # name the columns
index = c(1:length(adultcensus$age)) 
adultcensus = cbind(index, adultcensus) # create and bind an index column


## Cleaning Data and Splitting by Class ##

adultcensus = na.omit(adultcensus) # omit any rows with NA's
adultcensus = adultcensus %>% mutate(underover = ifelse(underover == levels(adultcensus$underover)[1], 0, 1)) # 0 is <=50k, 1 >50k # tunr into 1 or 0
adultcensus$underover = as.factor(adultcensus$underover) # turn back into factor
census.over = adultcensus[adultcensus$underover == 1,] # splitting data due to imbalance
census.under = adultcensus[adultcensus$underover == 0,]


## Under and Over sampling ##

under.balanced = census.under[sample(nrow(census.under), dim(census.over)[1]), ] # taking some of the rows from larger set 
adultcensus.undersampled = rbind.data.frame(census.over, under.balanced)
adultcensus.undersampled = adultcensus.undersampled[order(adultcensus.undersampled$index),] # ordering by index so no order with underover
over.balanced = census.over[sample(nrow(census.over), dim(census.under)[1], replace = TRUE), ] # taking repeated of the rows from smaller set 
adultcensus.oversampled = rbind.data.frame(census.under, over.balanced)
adultcensus.oversampled = adultcensus.undersampled[order(adultcensus.undersampled$index),] # ordering by index so no order with underover


## Undersampled ##

train.under = adultcensus.undersampled %>% sample_frac(0.8) # splitting into train and test sets
test.under = dplyr::anti_join(adultcensus.undersampled,train.under, by = 'index')


## Oversampled ##

train.over = adultcensus.oversampled %>% sample_frac(0.8) # splitting into train and test sets
test.over = dplyr::anti_join(adultcensus.oversampled,train.over, by = 'index')



'### VARIABLE IMPORTANCE ###'

## GLM and STEPWISE ##

model = glm(underover ~ ., family = binomial(link = 'logit'), data = train) # cretaes a glm model
model$xlevels[["native country"]] = union(model$xlevels[["native country"]], levels(test$`native country`)) # setting levles to be equal
predictions = predict.glm(model, test, type = 'response') #predicing test set
anova(model, test = 'Chisq')
model.step = step(model) # taking out unnecessary variables
summary(model.step) # show stepped model


## BORUTA feature selection ##
Boruta(adultcensus.undersampled, adultcensus.undersampled$underover) 
Boruta(adultcensus.oversampled, adultcensus.oversampled$underover)


'### MAKING ONE_HOT DATA MATRICES ###'

## one_hot undersampled ##

adultcensus.undersampled.lim = adultcensus.undersampled[, c(2, 3, 5, 7:16)] # taking the useful columns(remove index, fnlwgt, educ-num)
summary(adultcensus.undersampled.lim)
adultcensus.undersampled.lim$underover = as.numeric(adultcensus.undersampled.lim$underover) # turning to numeric
adultcensus.undersampled.lim = as.data.table(adultcensus.undersampled.lim) # has to be data.table for one hot encoding
undersamp.onehot = one_hot(adultcensus.undersampled.lim, cols = 'auto') # one hot encodes the data table
undersamp.labels = undersamp.onehot$underover # creates labels
undersamp.labels = ifelse(undersamp.labels == 1, 0, 1) # 0 is <=50k, 1 >50k
undersamp.onehot = as.matrix(undersamp.onehot[,-c(104)]) # removes labels from data frame
undersamp.onehot = na.omit(undersamp.onehot) # omits NA's made in one_hot


## one_hot oversampled ##

adultcensus.oversampled.lim = adultcensus.oversampled[, c(2, 3, 5, 7:16)] # taking the useful columns
summary(adultcensus.oversampled.lim)
adultcensus.oversampled.lim$underover = as.numeric(adultcensus.oversampled.lim$underover) # turning to numeric
adultcensus.oversampled.lim = as.data.table(adultcensus.oversampled.lim) # has to be data.table for one hot encoding
oversamp.onehot = one_hot(adultcensus.oversampled.lim, cols = 'auto') # one hot encodes the data table
oversamp.labels = oversamp.onehot$underover # creates labels
oversamp.labels = ifelse(oversamp.labels == 1, 0, 1) # 0 is <=50k, 1 >50k
oversamp.onehot = as.matrix(oversamp.onehot[,-c(104)]) # removes labels from data frame
oversamp.onehot = na.omit(oversamp.onehot) # omits NA's made in one_hot



'### KERAS CPU AND GPU CONFIGS ###'

## CPU ##

cpu = tf$config$experimental$list_physical_devices('CPU') # finds CPU
tf$config$experimental$set_visible_devices(cpu, 'CPU') # sets CPU as only physical device


## GPU ##

install_tensorflow(version = 'gpu') # grab gpu version, can use '1.14.0-gpu' to backdate
tf_config() # shows current config
gpus = tf$config$experimental$list_physical_devices('GPU') # shows all GPU devices
try(tf$config$experimental$set_memory_growth(device = gpus[[1]], enable = TRUE)) # takes first GPU device and stops it eating all the vram



'### KERAS MODEL ###'

## Building Models ##

build_model.nn = function() { # function to build the model
  
  model =- keras_model_sequential() %>% # building the model
    layer_dense(units = 128, activation = "relu",
                input_shape = dim(under.train)[2]) %>%
    layer_dense(units = 256, activation = "relu") %>%
    layer_dense(units = 128, activation = "relu") %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 1, activation = 'sigmoid')
  
  model %>% compile( # compiling the model
    loss = "binary_crossentropy",
    optimizer = optimizer_adam(lr = 0.01, beta_1 = 0.9, beta_2 = 0.999,
                               epsilon = NULL, decay = 0.000001, amsgrad = FALSE, clipnorm = NULL,
                               clipvalue = NULL), # shows custom optimiser
    metrics = list("accuracy")
  )
  
  model
}

print_dot_callback = callback_lambda( # Display progress by printing a single dot for each completed epoch.
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  }
)    

early_stop = callback_early_stopping(monitor = "val_loss", patience = 10) # early_stop makes it stop early when validation loss stops improving
## The patience parameter is the amount of epochs to check for improvement.


## Training Model ##

model.over = build_model.nn() # builds the model

# Fit the model and store training stats
history.over = model.over %>% fit(
  over.train,
  o.trn.labels,
  epochs = epochs,
  validation_split = 0.8,
  verbose = 0,
  callbacks = list(print_dot_callback)
)

plot(history, metrics = "accuracy", smooth = FALSE) + # plot accuracy over epoch
  coord_cartesian(ylim = c(0, 1))

model.over2 = build_model.nn() # create second model with early stopping
history = model.over2 %>% fit(
  over.train,
  o.trn.labels,
  epochs = epochs,
  validation_split = 0.8,
  verbose = 0,
  callbacks = list(print_dot_callback, early_stop)
)

plot(history, metrics = "accuracy", smooth = FALSE) +
  coord_cartesian(xlim = c(0, epochs), ylim = c(0, 1)) # plot accuracy over epoch



'### CARET MODEL ###'

# Grid of tuning parameters to try:
fitGrid = expand.grid(.size = c(2:5), .decay = c(0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7))

# Set the seeds for using parallel processing
# When parallel processing is used - set seeds for each iteration: 

set.seed(1)
seeds = vector(mode = "list", length = 11) # list of integer vectors that are used as seeds (number of resamples + 1) for final model
for(i in 1:10) seeds[[i]] = sample.int(n = 1000, 32) # the number of tuning parameter combinations is 4*8=32
seeds[[11]] = 1 # for the last model

# Set cross - validation, include seeds: 
fitControl = trainControl(method = "repeatedcv", 
                          number = 5,
                          repeats = 2,
                          classProbs = TRUE, 
                          summaryFunction =  twoClassSummary,
                          seeds = seeds # for sequential remove seeds
)

# Find out how many cores (workers) are available: 
detectCores()
cl = makeCluster(16)

# Register the workers from the cluster:
registerDoParallel(cl)

# Fit model using ANN, allow parallel computing: 
model.paral.nn = train(underover ~ .,
                       data = under.train,
                       method = "nnet",
                       maxit = 1000,
                       linout = FALSE,
                       trControl = fitControl,
                       tuneGrid = fitGrid,
                       trace = FALSE,
                       allowParallel = TRUE #for sequential remove allowParallel
)

# stopCluster will suspend the workers:
stopCluster(cl)

# registerDOSEQ will make the R environment single threaded:
registerDoSEQ()

model.paral.nn
plot(model.paral.nn)
varImp(model.ct.nn)



'### TEST AUC ###' 

o.preds = model.under2 %>% predict(over.test) # makes predictions
o.pred_obj = prediction(predictions = o.preds, labels = o.tst.labels) # turns into prediction object
performance(o.pred_obj, "auc")@y.values[[1]]