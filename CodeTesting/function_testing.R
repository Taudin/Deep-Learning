library(magick)
library(imager)
library(keras)

#Read in the asci_messy.pdf:
test_pdf <- image_read_pdf(path = "Testing/TestingData/asci_messy.pdf", density = 100)  #External pointer of class 'magick-image'.

#Convert to cimg by way of png:
test_cimg <- image_convert(image = test_pdf, format = "png") %>% magick2cimg()          #Large cimg.

#Crop out words surrounding relevant data area:
test_crop <- imsub(test_cimg, y > 284 & y < 1005, x > 279 & x < 600)                    #Large cimg.
#Create strips with one question per strip and stack strips along the z-axis:
test_strips <- imsplit(test_crop, "y", -36) %>% imappend(axis = "z")                    #Large cimg.
#Pad y-dimension to match the y-dimension size of the prepost strips (y = 46):
padded_test_strips <- test_strips %>% pad(10, axes = "y")                               #Large cimg.
#Make the padding black, so it isn't inadvertently trained by the model along with marks:
padded_test_strips[, c(1:5, 42:46),,] <- 1                                              #Large cimg.

#Convert to tensor format:
z_test <- dim(padded_test_strips)[3]
y_test <- dim(padded_test_strips)[2]
x_test <- dim(padded_test_strips)[1]
d_test <- dim(padded_test_strips)[4]
test_tnsr <- array(NA, dim = c(z_test, y_test, x_test, d_test))

#This is returned by the prep_asci() function:
test_list <- list(tnsr = test_tnsr, strips = padded_test_strips)

test_data1 <- test_list$tnsr                                                            #Large array.
test_data2 <- test_list$strips                                                          #Large cimg.
test_data3 <- test_list                                                                 #Large list.


test_strip_tnsr <- as.array(test_list$strips)
test_strip_tnsr <- array_reshape(test_strip_tnsr, dim = c(100, 46, 320, 3))

for (i in 1:z_test){
  test_tnsr[i,,,] <- frame(padded_test_strips, i)
}



# prep_labels() -----------------------------------------------------------


test_data_labels <- read.csv("Testing/TestingLabels/asci_messy_labels.csv")

test_labels <- test_data_labels[test_data_labels$survey == "asci",]

library(tidyr)
test_labels <- gather(test_labels, question, truth, q1:q20)

test_labels$category <- to_categorical(test_labels$truth, num_classes = 10)

plot(frame(padded_test_strips, 1))
test_labels$category[1,]
test_labels$truth[1]

plot(frame(padded_test_strips, 2))
test_labels$category[1,]
test_labels$truth[1]
