library(tidyverse)
library(keras)
library(magick)
library(dplyr)
library(stringr)
library(purrr)
library(tensorflow)

#load the CNN model and test data (the process was identical for training dataset)
test = read.csv("NewTest.csv")
model_cnn = load_model_hdf5("model_cnn_270320_2116_trunc.h5")

#Create function for image processing
process_img = function(imgx) {
  imgx = imgx %>%
    image_read() %>%
    image_scale("96x96!") %>%
    image_data("rgb") %>%
    as.integer()
  imgx / 255
}

#Preprocessing of images in test data
path_database = "C:/Users/Brian/Downloads/test_images/"
imgs_list = list.files(path_database)

#get pets IDs and the list of images associated to them
imgs_df = str_split(imgs_list, "-", simplify = T) %>%
  as.data.frame() %>%
  rename(PetID = V1) %>%
  select(PetID) %>%
  mutate(filename = imgs_list)

#create IDs for the images
imgs_df = inner_join(imgs_df, test) %>%
  filter(PetID %in% test$PetID) %>%
  mutate(imgID = str_remove(filename, ".jpg"))

#upload the images and process them
fnames_fullpath = paste0(path_database, imgs_df$filename)
x_test = map(fnames_fullpath, process_img)
x_test = simplify2array(x_test)
x_test = aperm(x_test, c(4,1,2,3))

#Applying model to entire test set
predictions = predict(model_cnn, x_test, checkpoint_path = NULL,
                      predict_keys = c("response"), hooks = NULL, as_iterable = FALSE,
                      simplify = TRUE, yield_single_examples = TRUE)

#Assign Pet IDs to associated images (for test predictions)
newtest = str_split(imgs_list, "-", simplify = T) %>%
  as.data.frame() %>%
  rename(PetID = V1) %>%
  select(PetID) %>%
  mutate(filename = imgs_list)

#Binding predictions for each pet to df
newtestbind = cbind(newtest, predictions)
newtestbind = newtestbind[,-2]

#Find mean image score for each pet in test
newtest1 = aggregate(predictions ~ PetID, newtestbind, mean)

#Exporting Image Prediction test File
write.csv(x=newtest1, file = "C:/Users/Brian/Documents/LJMU/SEMESTER 2/Data
Mining/Coursework/TestImageFeatures.csv", row.names = FALSE)