#Make sure you install the densenet package.
# Libraries ---------------------------------------------------------------
library(keras)
library(densenet)

# Parameters --------------------------------------------------------------

batch_size <- 64
epochs <- 300

# Data Preparation --------------------------------------------------------

#Load in the training and test data:
load("Code/training_and_test_data.Rdata")

# Normalisation
for(i in 1:3){
  mea <- mean(training_data[,,,i])
  sds <- sd(training_data[,,,i])
  
  training_data[,,,i] <- (training_data[,,,i] - mea) / sds
  test_data[,,,i] <- (test_data[,,,i] - mea) / sds
}
x_train <- training_data
x_test <- test_data


# Model Definition -------------------------------------------------------

input_img <- layer_input(shape = c(46, 320, 3))
  model <- application_densenet(include_top = TRUE, input_tensor = input_img, dropout_rate = 0.2)

opt <- optimizer_sgd(lr = 0.1, momentum = 0.9, nesterov = TRUE)

model %>% compile(
  optimizer = opt,
  loss = "categorical_crossentropy",
  metrics = "accuracy"
)

# Model fitting -----------------------------------------------------------

# callbacks for weights and learning rate
lr_schedule <- function(epoch, lr) {
  
  if(epoch <= 150) {
    0.1
  } else if(epoch > 150 && epoch <= 225){
    0.01
  } else {
    0.001
  }

}

lr_reducer <- callback_learning_rate_scheduler(lr_schedule)

history <- model %>% fit(
  training_data, training_labels, 
  batch_size = batch_size, 
  epochs = epochs, 
  validation_data = list(test_data, test_labels), 
  callbacks = list(
    lr_reducer
  )
)

plot(history)

evaluate(model, test_data, test_labels)