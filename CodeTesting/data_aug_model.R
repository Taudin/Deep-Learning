#Load keras with tensorflow backend with gpu support:
library(keras)
install_keras(tensorflow = "1.14.0-gpu")

#Load in the training and test data:
load("Code/training_and_test_data.Rdata")

#Instantiate a small convnet for survey response classification:
model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = c(46, 320, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten() %>% 
  layer_dense(units = 512, activation = "relu") %>% 
  layer_dense(units = 10, activation = "softmax")

#Look at how dimensions of the feature maps change with every successive layer:
summary(model)

#Configuring the model for training:
model %>% compile(
  loss = categorical_crossentropy,
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("acc")
)


# Setting up a data augmentation configuration via image_data_gene --------

datagen <- image_data_generator(
  rotation_range = 20,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

datagen %>% fit_image_data_generator(training_data)


# Fitting the model using a batch generator -------------------------------

history <- model %>% fit_generator(
  flow_images_from_data(training_data, training_labels, datagen, batch_size = 20),
  steps_per_epoch = 260,
  epochs = 40,
  validation_data = test_data
)


# Saving the model --------------------------------------------------------

model %>% save_model_hdf5("datagen_model2.h5")


# Displaying curves of loss and accuracy during training ------------------

plot(history)


# Displaying some randomly augmented training images ----------------------

augmentation_generator <- flow_images_from_data(training_data[1, 46, 320, 3]), generator = datagen, batch_size = 1) #Generates batches of randomly transformed images.

#Plot the images:
op <- par(mfrow = c(2, 2), pty = "s", mar = c(1, 0, 1, 0))
for 
