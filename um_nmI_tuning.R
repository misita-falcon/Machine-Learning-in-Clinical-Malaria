library(pacman)
library(plyr, dplyr)
library(ggplot2)
library(tidytree)
library(tensorflow)
library(keras)
library(tidyr, caret)
library(yardstick)
library(recipes)
library(rsample)

pythonpath='/opt/apps/Python/Python-3.6.8/bin/python3'
Sys.setenv(RETICULATE_PYTHON=pythonpath)
library(reticulate)
library(keras)
library(tensorflow)
#confirm
reticulate::py_discover_config()

#get preprocessed data
clinhem <- read.csv("/home/cmmoranga/machine_learn/um/Imputed_um_nmi.csv", header = T, na.strings = T) ; glimpse(clinhem)

#######################################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#remove the X column, hb_level, hematocrit
clinhem <- clinhem %>% select(-c(X, hb_level, hematocrit,location, patient_age)); glimpse(clinhem)
#randomize the data 
clinhem <- clinhem[sample(1:nrow(clinhem)), ]
# Split test/training sets
set.seed(1234)
train_test_split <- initial_split(clinhem, prop = 0.8); train_test_split
## Retrieve train and test sets
train_tbl_with_ids <- training(train_test_split); test_tbl_with_ids  <- testing(train_test_split)
train_tbl <- select(train_tbl_with_ids, -SampleID); test_tbl <- select(test_tbl_with_ids, -SampleID)
# Create recipe
recipe_UM <- recipe(Clinical_Diagnosis ~ ., data = train_tbl) %>%
  #step_dummy(all_nominal(), -all_outcomes()) %>%
  step_YeoJohnson(all_predictors(), -all_outcomes()) %>%
  step_center(all_predictors(), -all_outcomes()) %>%
  step_scale(all_predictors(), -all_outcomes()) %>%
  prep(data = train_tbl)
recipe_UM
# Predictors
x_train_tbl2 <- bake(recipe_UM, new_data = train_tbl) ; x_test_tbl2  <- bake(recipe_UM, new_data = test_tbl)
x_train_tbl <- x_train_tbl2 %>% select(-Clinical_Diagnosis) ; x_test_tbl <- x_test_tbl2 %>% select(-Clinical_Diagnosis)
# Response variables for training and testing sets
y_train_vec <- ifelse(pull(train_tbl, Clinical_Diagnosis) == "Uncomplicated Malaria", 1, 0)
y_test_vec  <- ifelse(pull(test_tbl, Clinical_Diagnosis) == "Uncomplicated Malaria", 1, 0)

######################################################################################################################

# Building our Artificial Neural 
# Hyperparameter flags ---------------------------------------------------

FLAGS <- flags(
  # nodes
  flag_numeric("nodes1", 256),
  flag_numeric("nodes2", 128),
  flag_numeric("nodes3", 64),
  # dropout
  flag_numeric("dropout1", 0.4),
  flag_numeric("dropout2", 0.3),
  flag_numeric("dropout3", 0.2),
  # learning paramaters
  flag_string("optimizer", "adam"),
  flag_numeric("lr_annealing", 0.1)
)

# Define Model --------------------------------------------------------------

model <- keras_model_sequential() %>%
  layer_dense(units = FLAGS$nodes1, activation = "relu", input_shape = ncol(x_train_tbl)) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = FLAGS$dropout1) %>%
  layer_dense(units = FLAGS$nodes2, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = FLAGS$dropout2) %>%
  layer_dense(units = FLAGS$nodes3, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = FLAGS$dropout3) %>%
  layer_dense(units = 1, activation = "sigmoid") %>%
  compile(
    loss = 'binary_crossentropy',
    metrics = c('accuracy'),
    optimizer = FLAGS$optimizer
  ) %>%
  fit(
    x = as.matrix(x_train_tbl),
    y = y_train_vec,
    epochs = 100,
    batch_size = 64,
    validation_split = 0.3,
    callbacks = list(
      callback_early_stopping(patience = 5),
      callback_reduce_lr_on_plateau(factor = FLAGS$lr_annealing)
    ),
    verbose = FALSE
  )



