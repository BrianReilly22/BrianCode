library(keras)
library(tensorflow)
library(magick)
library(tidyverse)
library(dplyr)
library(stringr)
library(purrr)

#Read in image data and tabular data
path_database = "C:/Users/aribreil/Downloads/"
path_database_images = paste0(path_database, "train_images/")
train = read.csv(paste0(path_database, "NewTrain.csv"))

#Put the images in memory
imgs_list = list.files(path_database_images)

#Assign petIDs to associated images
imgs_df = str_split(imgs_list, "-", simplify = T) %>%
  as.data.frame() %>%
  rename(PetID = V1) %>%
  select(PetID) %>%
  mutate(filename = imgs_list)

#Create image IDs
imgs_df = inner_join(imgs_df, train) %>%
  filter(PetID %in% train$PetID) %>%
  mutate(imgID = str_remove(filename, ".jpg"))
imgs_df = imgs_df %>%
  mutate(Adopted = ifelse(Adopted == "NO", 0,1))

#Split training to (mini)train and (mini)test
indx_sample = sample.int(nrow(imgs_df))
N_train = round(length(indx_sample) * 0.8 )
N_test = nrow(imgs_df) - N_train

#And retain total
N_total = round(length(indx_sample) * 1.0 )

#Create function for image processing
process_img = function(imgx) {
  imgx = imgx %>%
    image_read() %>%
    image_scale("50x50!") %>%
    image_data("rgb") %>%
    as.integer()
  imgx / 255
}

#Apply function to training and test set
fnames_fullpath = paste0(path_database_images, imgs_df$filename)
x_train = map(fnames_fullpath[indx_sample[1:N_train]], process_img)
x_train = simplify2array(x_train)

x_train = aperm(x_train, c(4,1,2,3))
x_test = map(fnames_fullpath[indx_sample[(N_train+1):(N_train + N_test)]], process_img)
x_test = simplify2array(x_test)
x_test = aperm(x_test, c(4,1,2,3))

#For total training data
x_total = map(fnames_fullpath[indx_sample[1:N_total]], process_img)
x_total = simplify2array(x_total)
x_total = aperm(x_total, c(4,1,2,3))

#Keep the class lables separated in vectors
y_train = imgs_df$Adopted[indx_sample[1:N_train]]
y_test = imgs_df$Adopted[indx_sample[(N_train+1):(N_train + N_test)]]

#Building CNN model
build_model = function() {
  model = keras_model_sequential()
  model %>%
    layer_conv_2d(
      filter = 32, kernel_size = c(3,3), padding = "same",
      input_shape = c(50, 50, 3)
    ) %>%
    layer_activation("relu") %>%
    # Use max pooling
    layer_max_pooling_2d(pool_size = c(2,2)) %>%
    #Apply dropout
    layer_dropout(0.3) %>%
    layer_conv_2d(
      filter = 16, kernel_size = c(3,3), padding = "same"
    ) %>%
    layer_activation("relu") %>%
    # Use max pooling
    layer_max_pooling_2d(pool_size = c(2,2)) %>%
    #Apply dropout
    layer_dropout(0.3) %>%
    # Flatten max filtered output into feature vector
    # and feed into dense layer
    layer_flatten() %>%
    layer_dense(128) %>%
    layer_activation("relu") %>%
    layer_dropout(0.3) %>%
    layer_dense(1) %>%
    layer_activation("sigmoid")
  model %>% compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
    metrics = "accuracy"
  )
  model
}

#Actually building the function
model = build_model()
model

#Model Training
batch_size = 128
epochs = 30
history = model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  validation_split = 0.2,
  shuffle = TRUE
)

#Evaluating model perfomance
model %>% evaluate(x_test, y_test)
#Applying model to entire training set
predictions = predict(model, x_total, checkpoint_path = NULL,
                      predict_keys = c("response"), hooks = NULL, as_iterable = FALSE,
                      simplify = TRUE, yield_single_examples = TRUE)

#Assign Pet IDs to associated images
new = str_split(imgs_list, "-", simplify = T) %>%
  as.data.frame() %>%
  rename(PetID = V1) %>%
  select(PetID) %>%
  mutate(filename = imgs_list)

#Binding predictions for each pet to df
new1 = cbind(new, predictions)
new1 = new1[,-2]

#Find mean image score for each pet in training
new1 = aggregate(predictions ~ PetID, new1, mean)
names(new1)[2] <- "ImagePrediction"

#Merge to training data
TrainNew = merge(train, new1,by="PetID")
summary(TrainNew)

#Pre processing of images in test data, same as for train
path_database_images1 = paste0(path_database, "test_images/")
test = read.csv(paste0(path_database, "NewTest.csv"))
imgs_list1 = list.files(path_database_images1)
imgs_df1 = str_split(imgs_list1, "-", simplify = T) %>%
  as.data.frame() %>%
  rename(PetID = V1) %>%
  select(PetID) %>%
  mutate(filename = imgs_list1)
imgs_df1 = inner_join(imgs_df1, test) %>%
  filter(PetID %in% test$PetID) %>%
  mutate(imgID = str_remove(filename, ".jpg"))

indx_sample1 = sample.int(nrow(imgs_df1))
N_total1 = round(length(indx_sample1) * 1.0 )
fnames_fullpath1 = paste0(path_database_images1, imgs_df1$filename)
x_total1 = map(fnames_fullpath1[indx_sample1[1:N_total1]], process_img)
x_total1 = simplify2array(x_total1)
x_total1 = aperm(x_total1, c(4,1,2,3))

#Applying model to entire test set
predictions1 = predict(model, x_total1, checkpoint_path = NULL,
                       predict_keys = c("response"), hooks = NULL, as_iterable = FALSE,
                       simplify = TRUE, yield_single_examples = TRUE)

#Assign Pet IDs to associated images (for test predictions)
newtest = str_split(imgs_list1, "-", simplify = T) %>%
  as.data.frame() %>%
  rename(PetID = V1) %>%
  select(PetID) %>%
  mutate(filename = imgs_list1)

#Binding predictions for each pet to df
newtest1 = cbind(newtest, predictions1)
newtest1 = newtest1[,-2]

#Find mean image score for each pet in test
newtest1 = aggregate(predictions1 ~ PetID, newtest1, mean)
names(newtest1)[2] <- "ImagePrediction"

#Merge to test data
TestNew = merge(test, newtest1,by="PetID")
summary(TestNew)

#Exporting New Training Data set
write_xlsx(x = TrainNew, path = "TrainNew.xlsx", col_names = TRUE)

#Exporting New Test Data set
write_xlsx(x = TestNew, path = "TestNew.xlsx", col_names = TRUE)