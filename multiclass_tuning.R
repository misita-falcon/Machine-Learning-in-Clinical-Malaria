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
clinhem <- read.csv("/home/cmmoranga/machine_learn/multi/Imputed_um_sm_nmi_Age.csv", header = T, na.strings = T) ; glimpse(clinhem)

#mutate factors to characters
clinhem %>% mutate_if(is.factor, as.character) -> clinhem ;
#encode the clinical diagnosis
#######################################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#encode the clinical diagnosis
clinhem$Clinical_Diagnosis2 <- NULL
clinhem$Clinical_Diagnosis2[clinhem$Clinical_Diagnosis == "Non-malaria Infection"] = 0
clinhem$Clinical_Diagnosis2[clinhem$Clinical_Diagnosis == "Uncomplicated Malaria"] = 1
clinhem$Clinical_Diagnosis2[clinhem$Clinical_Diagnosis == "Severe Malaria"] = 2

#######################################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#remove the first column, hb_level, Clinical_Diagnosis,location, patient_age, hematocrit
clinhem <- clinhem %>% select(-c(X, hb_level, Clinical_Diagnosis,location, patient_age, hematocrit)); glimpse(clinhem)
#randomize the data 
clinhem <- clinhem[sample(1:nrow(clinhem)), ]
# Split test/training sets
set.seed(1000)
train_test_split <- initial_split(clinhem, prop = 0.8); train_test_split
## Retrieve train and test sets
train_tbl_with_ids <- training(train_test_split); test_tbl_with_ids  <- testing(train_test_split)
#create dataset with 
train_tbl <- select(train_tbl_with_ids, -SampleID); test_tbl <- select(test_tbl_with_ids, -SampleID)
# Create recipe
recipe_UM <- recipe(Clinical_Diagnosis2 ~ ., data = train_tbl) %>%
  step_YeoJohnson(all_predictors(), -all_outcomes()) %>%
  step_center(all_predictors(), -all_outcomes()) %>%
  step_scale(all_predictors(), -all_outcomes()) %>%
  prep(data = train_tbl)
recipe_UM
# Predictors
x_train_tbl2 <- bake(recipe_UM, new_data = train_tbl); x_test_tbl2  <- bake(recipe_UM, new_data = test_tbl)
#remove the outcome from the data
x_train_tbl <- x_train_tbl2 %>% select(-Clinical_Diagnosis2); x_test_tbl <- x_test_tbl2 %>% select(-Clinical_Diagnosis2)
# Response variables for training and testing sets
y_train_vec <- to_categorical(train_tbl$Clinical_Diagnosis2) ; y_test_vec  <- to_categorical(test_tbl$Clinical_Diagnosis2); table(y_train_vec) ; table(y_test_vec)

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
  layer_dense(units = 3, activation = "softmax") %>%
  compile(
    loss = 'categorical_crossentropy',
    metrics = c('accuracy'),
    optimizer = FLAGS$optimizer
  ) %>%
  fit(
    x = as.matrix(x_train_tbl),
    y = y_train_vec,
    epochs = 90,
    batch_size = 64,
    validation_split = 0.3,
    callbacks = list(
      callback_early_stopping(patience = 5),
      callback_reduce_lr_on_plateau(factor = FLAGS$lr_annealing)
    ),
    verbose = FALSE
  )



