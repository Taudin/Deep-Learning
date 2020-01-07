# Pre-processing ----------------------------------------------------------

#Load necessary packages:
library(imager)
library(magick)
library(keras)
install_keras(method = "virtualenv", conda = "auto", version = "default", tensorflow = "gpu")
library(tidyr)
library(pdftools)
library(abind)
#Load necessary strip and label functions:
source("Code/strip_functions.R")


# Only One Survey Type ----------------------------------------------------

#Get strips and tensor objects created from PDFs:
asci_mixed_data <- prep_asci(image_file = "Training/TrainingData/asci_mixed.pdf", fromfile = TRUE, type = "scan")$tnsr

#Create test data:
test_data <- prep_asci(image_file = "Testing/TestingData/asci_messy.pdf", fromfile = TRUE, type = "scan")$tnsr

#Import the corresponding csv file for asci_mixed.pdf:
asci_mixed_labels <- read.csv("Training/TrainingLabels/asci_mixed_labels.csv")
#Prepare the asci_mixed labels for the first group of training labels:
asci_mixed_training_labels <- prep_labels(label_data = asci_mixed_labels, survey_name = "asci")$category

#Import the corresponding csv file for asci_messy.pdf:
asci_messy_labels <- read.csv("Testing/TestingLabels/asci_messy_labels.csv")
#Prepare the test labels for the test_data:
test_labels <- prep_labels(label_data = asci_messy_labels, survey_name = "asci")$category


# 4 page Combined Surveys -------------------------------------------------

#Get the path to the 4 page combined survey PDF:
file <- "Training/TrainingData/4pg_marked_combo.pdf"

#Index the appropriate survey for separation:
num_pages <- pdf_info(file)$pages
asci_pages <- seq(1, num_pages, by = 4)     #A vector of indices that correspond to the ASCI surveys (which are the first page of the group).

#Extract the ASCI surveys from the 4 page combined survey PDF:
asci_pdf <- image_read_pdf(file, density = 100, pages = asci_pages)
asci_cimg <- image_convert(asci_pdf, "png") %>% magick2cimg()
asci_data <- prep_asci(image_file = asci_cimg, fromfile = FALSE, type = "4pg_combined_scan")$tnsr

#Import the csv file for the 4pg_marked_combo.pdf:
four_page_labels <- read.csv("Training/TrainingLabels/4pg_labels.csv")

#Get the labels that correspond to the extracted ASCI PDFs from 4pg_marked_combo.pdf:
asci_extract_training_labels <- prep_labels(label_data = four_page_labels, survey_name = "asci")$category



# Create Training Data -------------------------------------------

#Form training data:
training_data <- abind(asci_mixed_data, asci_data, along = 1)

#Form training labels:
training_labels <- abind(asci_mixed_training_labels, asci_extract_training_labels, along = 1)


# Save as R Workspace -----------------------------------------------------

rm(list = ls()[!ls() %in% c("training_data", "training_labels", "test_data", "test_labels")])
gc()
base::save.image("Code/training_and_test_data.Rdata")



