#Load necessary packages:
library(imager)
library(magick)
library(keras)
install_keras(tensorflow = "1.14.0-gpu")
library(tidyr)

#############################################################################################################################################
#                                                                                                                                           #
#                                                             prep_asci()                                                                   #
#                                                                                                                                           #
#                                                                                                                                           #
# This function can read in ASCI surveys, whether they are the stand-alone ASCI PDFs located in a directory on disk or they are part of the #
# 4 page combined surveys (in which they are the first page) and already converted to cimg by the preprocess.R script. This function        #
# returns tensor objects for the model after slicing the survey into one strip per item. If reading in from PDF set image_file = "path to   #
# location on disk" and fromfile = TRUE. If this function is being called from another file where the PDF has already been converted into a #
# cimg, then set image_file = object_name, and fromfile = FALSE.                                                                            #
#############################################################################################################################################
prep_asci <- function(image_file, fromfile = FALSE, type = "scan"){
  #If the ASCI PDF has been scanned in and is stored in a directory, e.g., non-combined surveys that are all ASCI PDFs:
  if (fromfile){
    #Read in the PDF:
    image_pdf <- image_read_pdf(image_file, density = 100)
    #Convert to cimg by way of png:
    png_cimg <- image_convert(image_pdf, "png") %>% magick2cimg()
  } else {
    #Doesn't need the above conversion since it's already been converted into a cimg, e.g., this ASCI survey PDF a part of the 4 page 
    #combined surveys.
    png_cimg <- image_file
  }
  
  #If the ASCI PDF has been scanned in and was stored in a directory-- a continuation of the above if statement codeblock...
  if (type == "scan"){
    #Crop out words surrounding relevant data area:
    image_crop <- imsub(png_cimg, y > 284 & y < 1005, x > 279 & x < 600)
    #Create strips with one question per strip and stack strips along the z-axis:
    strips <- imsplit(image_crop, "y", -36) %>% imappend(axis = "z")
    #Pad y-dimension to match the y-dimension size of the prepost strips (y = 46):
    padded_strips <- strips %>% pad(10, axes = "y")
    #Make the padding black, so it isn't inadvertently trained by the model along with marks:
    padded_strips[, c(1:5, 42:46),,] <- 1
  }
  
  #If this is the ASCI survey part from a 4 page combined survey...
  if (type %in% c("pdf", "4pg_combined_scan")){
    #Crop out words surrounding relevant data area:
    image_crop <- imsub(png_cimg, y > 264 & y < 1025, x > 279 & x < 600)
    #Create strips with one question per strip and stack strips along z-axis:
    strips <- imsplit(image_crop, "y", -38) %>% imappend(axis = "z")
    #Pad y-dimension to match the y-dimension size of the prepost strips (y = 46):
    padded_strips <- strips %>% pad(8, axes = "y")
    #Make the padding black, so it isn't inadvertently trained by the model along with marks:
    padded_strips[, c(1:4, 43:46),,] <- 1
  }
  
  #Convert to tensor format:
  z <- dim(padded_strips)[3]
  y <- dim(padded_strips)[2]
  x <- dim(padded_strips)[1]
  d <- dim(padded_strips)[4]
  tnsr <- array(NA, dim = c(z, x, y, d))

  #Return the tensor object:
  return(list(tnsr = tnsr, strips = as.array(padded_strips)))
}


############################################################################################################################################
#                                                                                                                                          #
#                                                            prep_prepost()                                                                #
#                                                                                                                                          #
# This function reads in the prepost survey whether it is a stand-alone PDF or part of the 4 page combined surveys, makes strips out of    #
# each item, and returns it as a tensor object to be used in a model. It can read in prepost survey PDFs whether they come from a          #
# directory stored on disk or already converted into cimg objects from preprocessing.R.                                                    #
############################################################################################################################################
prep_prepost <- function(image_file, fromfile = FALSE){
  #If this is a stand-alone prepost survey PDF:
  if (fromfile){
    survey_pdf <- image_read_pdf(image_file, density = 72)
    png_cimg <- image_convert(survey_pdf, "png") %>% magick2cimg()
  } else {
    #This prepost survey was a part of the 4 page combined surveys...
    png_cimg <- image_file     #Assign to the generic variable name for cimg's in this function.
  }
  
  num_pages <- dim(png_cimg)[3]
  
  #Trim excess, including left and right words:
  first_page <- imsub(frame(png_cimg, seq(1, num_pages, 2)), y > 184 & y < 737, x > 200 & x < 451)
  second_page <- imsub(frame(png_cimg, seq(2, num_pages, 2)), y > 61 & y < 430, x > 200 & x < 451)
  
  #Make strips with one go and stack along the z-axis:
  first_page_long <- imsplit(first_page, axis = "y", -46) %>% imappend(axis = "z")
  second_page_long <- imsplit(second_page, axis = "y", -46) %>% imappend(axis = "z")
  all_pages_long <- imappend(list(first_page_long, second_page_long), axis = "z")
  
  #Pad along the x-axis so we can merge along the x-axis:
  padded_strips <- all_pages_long %>% pad(70, axes = "x")
  #Make the padding black, so it isn't inadvertently trained by the model along with marks:
  padded_strips[c(1:35, 286:320),,,] <- 1
  
  #Convert to tensor object:
  z <- dim(padded_strips)[3]
  y <- dim(padded_strips)[2]
  x <- dim(padded_strips)[1]
  d <- dim(padded_strips)[4]
  tnsr <- array(NA, dim = c(z, y, x, d))
  
  #Return the tensor object:
  return(list(tnsr = tnsr, strips = padded_strips))
}


###########################################################################################################################################
#                                                                                                                                         #                                                           
#                                                          prep_labels()                                                                  #
#                                                                                                                                         #
# This function takes in a read csv file, matches up the labels to that particular PDF, categorizes the labels corresponding to the items #
# answered in the PDF survey, and returns those labels to preprocess.R.                                                                   #
###########################################################################################################################################
prep_labels <- function(label_data, survey_name){
  #Match up the particular data from the csv survey rows to the appropriate PDF file:
  labels <- label_data[label_data$survey == survey_name,]
  #Gather takes multiple columns and collapses into key = question, value = truth pairs. Basically, the columns q1-q20 are turned into 
  #observations --  each item answer becomes an observation with the variable name question (q1-q20) and truth (the corresponding answer)
  #as a dataframe object.
  labels <- gather(labels, question, truth, q1:q20)
  #One-hot encode each observation's answer as a new variable named category:
  labels$category <- to_categorical(labels$truth, num_classes = 10)
  
  #Return the labels:
  return(labels)
}

