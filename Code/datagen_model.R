library(keras)
install_keras(tensorflow = "1.14.0-gpu")

# Parameters --------------------------------------------------------------

batch_size <- 32
epochs <- 200
data_augmentation <- TRUE


# Data Preparation --------------------------------------------------------

#Load in the training and test data:
load("Code/training_and_test_data.Rdata")

# Defining Model ----------------------------------------------------------

# Initialize sequential model
model <- keras_model_sequential()

model %>%
   # Start with hidden 2D convolutional layer being fed 32x32 pixel images
  layer_conv_2d(
    filter = 32, kernel_size = c(3,3), padding = "same", 
    input_shape = c(46, 320, 3)
  ) %>%
  layer_activation("relu") %>%
  # Second hidden layer
  layer_conv_2d(filter = 32, kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  # 2 additional hidden 2D convolutional layers
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same") %>%
  layer_activation("relu") %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%
  # Use max pooling once more
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  # Flatten max filtered output into feature vector 
  # and feed into dense layer
  layer_flatten() %>%
  layer_dense(512) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%
  # Outputs from dense layer are projected onto 10 unit output layer
  layer_dense(10) %>%
  layer_activation("softmax")

opt <- optimizer_rmsprop(lr = 0.0001, decay = 1e-6)

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = opt,
  metrics = "accuracy"
)


# Training ----------------------------------------------------------------

if(!data_augmentation){
  
  model %>% fit(
    training_data, training_labels,
    batch_size = batch_size,
    epochs = epochs,
    validation_data = list(test_data, test_labels),
    shuffle = TRUE
  )
  
} else {
  
  datagen <- image_data_generator(
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    horizontal_flip = TRUE
  )
  
  datagen %>% fit_image_data_generator(training_data)
  
  model %>% fit_generator(
    flow_images_from_data(training_data, training_labels, datagen, batch_size = batch_size),
    steps_per_epoch = as.integer(50000/batch_size), 
    epochs = epochs, 
    validation_data = list(test_data, test_labels)
  )
  
}