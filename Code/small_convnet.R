# Instantiating a small convnet -------------------------------------------
#Load packages:
library(keras)
library(magick)
library(imager)
library(pdftools)

model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(46, 320, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu")

#Display the architecture of the convnet:
model


# Adding a classifier on top of the convnet -------------------------------

model <- model %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 10, activation = "softmax")

#Here's what the network looks like now:
model


# Training the convnet on the ASCI PDFs -----------------------------------

#Read in the ASCI mixed mark PDF named asci_mixed.pdf:
asci_mixed_pdf <- image_read_pdf(path = "Training/TrainingData/asci_mixed.pdf", density = 100)
#Convert the external pointer of class 'magick-image' to a cimg:
asci_mixed_png_cimg <- image_convert(asci_mixed_pdf, "png") %>% magick2cimg()
#Crop outer edges of the cimg:
asci_mixed_image_crop <- imsub(asci_mixed_png_cimg, y > 284 & y < 1005, x > 279 & x < 600)
#Create strips with one question per strip and stack strips along the z-axis:
asci_mixed_strips <- imsplit(asci_mixed_image_crop, "y", -36) %>% imappend(axis = "z")
#Pad y-dimension to match the y-dimension size of the prepost strips (y = 46):
asci_mixed_padded_strips <- asci_mixed_strips %>% pad(10, axes = "y")
#Make the padding black, so it isn't inadvertently trained by the model along with marks:
asci_mixed_padded_strips[, c(1:5, 42:46),,] <- 1

#Convert to tensor format:
mixed_z <- dim(asci_mixed_padded_strips)[3]
mixed_y <- dim(asci_mixed_padded_strips)[2]
mixed_x <- dim(asci_mixed_padded_strips)[1]
mixed_d <- dim(asci_mixed_padded_strips)[4]
asci_mixed_tnsr <- array(NA, dim = c(mixed_z, mixed_y, mixed_x, mixed_d))

for (i in 1:mixed_z){
  asci_mixed_tnsr[i,,,] <- frame(asci_mixed_padded_strips, i)
}


#Get the path to the 4 page combined survey PDF:
file <- "Training/TrainingData/4pg_marked_combo.pdf"
#Index the appropriate survey for separation:
num_pages <- pdf_info(file)$pages
asci_pages <- seq(1, num_pages, by = 4)     #A vector of indices that correspond to the ASCI surveys (which are the first page of the group).

#Extract the ASCI surveys from the 4 page combined survey PDF:
asci_pdf <- image_read_pdf(file, density = 100, pages = asci_pages)
asci_cimg <- image_convert(asci_pdf, "png") %>% magick2cimg()
asci_image_crop <- imsub(asci_cimg, y > 264 & y < 1025, x > 279 & x < 600)
#Create strips with one question per strip and stack strips along z-axis:
asci_strips <- imsplit(asci_image_crop, "y", -38) %>% imappend(axis = "z")
#Pad y-dimension to match the y-dimension size of the prepost strips (y = 46):
asci_padded_strips <- asci_strips %>% pad(8, axes = "y")
#Make the padding black, so it isn't inadvertently trained by the model along with marks:
asci_padded_strips[, c(1:4, 43:46),,] <- 1

#Convert to tensor format:
asci_z <- dim(asci_padded_strips)[3]
asci_y <- dim(asci_padded_strips)[2]
asci_x <- dim(asci_padded_strips)[1]
asci_d <- dim(asci_padded_strips)[4]
asci_tnsr <- array(NA, dim = c(asci_z, asci_y, asci_x, asci_d))

for (i in 1:asci_z){
  asci_tnsr[i,,,] <- frame(asci_padded_strips, i)
}

training_data <- abind(asci_mixed_tnsr, asci_tnsr, along = 1)


#Import the corresponding csv file for asci_mixed.pdf:
asci_mixed_labels <- read.csv("Training/TrainingLabels/asci_mixed_labels.csv")
asci_mixed_labels <- gather(asci_mixed_labels, question, truth, q1:q20)
asci_mixed_labels$category <- to_categorical(asci_mixed_labels$truth, num_classes = 10)
asci_mixed_training_labels <- asci_mixed_labels$category

asci_labels <- read.csv("Training/TrainingLabels/4pg_labels.csv")
asci_labels <- asci_labels[asci_labels$survey == "asci",]
asci_labels <- gather(asci_labels, question, truth, q1:q20)
asci_labels$category <- to_categorical(asci_labels$truth, num_classes = 10)
asci_training_labels <- asci_labels$category

training_labels <- abind(asci_mixed_training_labels, asci_training_labels, along = 1)




test_asci <- image_read_pdf(path = "Testing/TestingData/asci_messy.pdf", density = 100)
#Convert the external pointer of class 'magick-image' to a cimg:
test_asci_cimg <- image_convert(test_asci, "png") %>% magick2cimg()
#Crop outer edges of the cimg:
test_asci_crop <- imsub(test_asci_cimg, y > 284 & y < 1005, x > 279 & x < 600)
#Create strips with one question per strip and stack strips along the z-axis:
test_asci_strips <- imsplit(test_asci_crop, "y", -36) %>% imappend(axis = "z")
#Pad y-dimension to match the y-dimension size of the prepost strips (y = 46):
test_asci_padded_strips <- test_asci_strips %>% pad(10, axes = "y")
#Make the padding black, so it isn't inadvertently trained by the model along with marks:
test_asci_padded_strips[, c(1:5, 42:46),,] <- 1

#Convert to tensor format:
test_asci_z <- dim(test_asci_padded_strips)[3]
test_asci_y <- dim(test_asci_padded_strips)[2]
test_asci_x <- dim(test_asci_padded_strips)[1]
test_asci_d <- dim(test_asci_padded_strips)[4]
test_asci_tnsr <- array(NA, dim = c(test_asci_z, test_asci_y, test_asci_x, test_asci_d))

for (i in 1:test_asci_z){
  test_asci_tnsr[i,,,] <- frame(test_asci_padded_strips, i)
}

test_data <- test_asci_tnsr



#Import the corresponding csv file for asci_mixed.pdf:
asci_test_labels <- read.csv("Testing/TestingLabels/asci_messy_labels.csv")
test_asci_labels <- gather(asci_test_labels, question, truth, q1:q20)
test_asci_labels$category <- to_categorical(test_asci_labels$truth, num_classes = 10)
test_labels1 <- test_asci_labels$category
test_labels2 <- test_asci_labels$truth

model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

model %>% fit(
  training_data, training_labels,
  epochs = 20, batch_size = 5
)

#Evaluate the model on the test set:
results <- model %>% evaluate(test_data, test_labels1)
results

class_predict <- model %>% predict_classes(test_data)
mean(test_labels2 == as.vector(class_predict))
table(test_labels2, class_predict)


