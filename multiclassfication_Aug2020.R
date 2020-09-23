#Script for ANN for UM vs nMI
####################################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(pacman)
pacman::p_load(ggplot2, reshape2, gplots, grid, spatstat, raster, sp, dplyr, 
                klaR, ggfortify, stringr, cluster, Rtsne, readr, RColorBrewer, Hmisc, mice, tidyr, 
                purrr, VIM, magrittr, corrplot, caret, gridExtra, ape, tidytree, pheatmap, stats, 
                vegan, FactoMineR, factoextra, outliers, ggpubr, keras, lime, tidyquant, rsample, 
                recipes, corrr, yardstick, tensorflow, caret, limma, compareGroups, forcats)
#S####################################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
setwd("/home/root01/Documents/machine_learn_sep_2020/Analysis/")

#get pre-processed data
clinhem <- read.csv("../Data/Imputed_um_sm_nmi_Age.csv", header = T, na.strings = T); glimpse(clinhem)
#remove the X column, hb_level, hematocrit
#mutate factors to characters
clinhem %>% mutate_if(is.factor, as.character) -> clinhem ;
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

#######################################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Building our Artificial Neural Network
model_keras <- keras_model_sequential()
model_keras %>%
  # First hidden layer and Dropout to prevent overfitting
layer_dense(units = 128, kernel_initializer = "uniform", activation = "relu", 
            input_shape = ncol(x_train_tbl)) %>% 
  layer_dropout(rate = 0.4) %>% layer_batch_normalization() %>%
  # Second hidden layer and Dropout to prevent overfitting
layer_dense(units = 64, kernel_initializer = "uniform", activation= "relu") %>%
  layer_dropout(rate = 0.3) %>% layer_batch_normalization() %>%
layer_dense(units = 16, kernel_initializer = "uniform", activation= "relu") %>%
  layer_dropout(rate = 0.4) %>% layer_batch_normalization() %>%
  # Output layer
layer_dense(units= 3, kernel_initializer = "uniform", activation = "softmax") %>%
compile( loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')); model_keras
# Fit the keras model to the training data
set.seed(1234)
fit_keras <- fit(
  object           = model_keras, 
  x                = as.matrix(x_train_tbl), 
  y                = y_train_vec,
  batch_size       = 64, 
  epochs           = 500,
  validation_split = 0.30, #for cross validation
  shuffle          = TRUE, verbose  = TRUE,
  callbacks = list(
    #callback_early_stopping(patience = 50),
    callback_tensorboard("multiclassfication_may"),
    callback_reduce_lr_on_plateau(factor = 0.0001))
  ); tensorboard("multiclassfication_may"); 
# Print the final model
fit_keras ; save_model_hdf5(model_keras, 'sm_um_nmf.hdf5')
# Plot the training/validation history of our Keras model
plot_keras <- plot(fit_keras) +
  theme_tq() +
  scale_color_tq() +
  scale_fill_tq() 
#labs(title = "Accuracy and loss of during Training for Severe malaria") ;
plot_keras; wplot_save_this("sm_um_nmf")

#######################################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Predicted Class
yhat_keras_class_vec <- predict_classes(object = model_keras, x = as.matrix(x_test_tbl)) %>%
  as.vector()
# Predicted Class Probability
yhat_keras_prob_vec  <- predict_proba(object = model_keras, x = as.matrix(x_test_tbl)) %>%
  as.vector()
#Reverse the categorical variable (one hot encode)
inverse_to_categorical <- function(mat)
{
  apply(mat, 1, function(row) which(row==max(row))-1)
}
y_test_vec2 <- inverse_to_categorical(y_test_vec)
# Format test data and predictions for yardstick metrics
estimates_keras_tbl <- tibble(
  truth      = as.factor(y_test_vec2) %>% fct_recode(nMI = "0", UM = "1", SM = "2"),
  estimate   = as.factor(yhat_keras_class_vec) %>% fct_recode(nMI = "0", UM = "1", SM = "2"),
  class_prob = yhat_keras_class_vec 
)
#evaluate the predictions
estimates_keras_tbl
options(yardstick.event_first = FALSE)
# Confusion Table
estimates_keras_tbl %>% conf_mat(truth, estimate)
# Accuracy
estimates_keras_tbl %>% metrics(truth, estimate)
# Precision
estimates_keras_tbl %>% precision(truth, estimate)
estimates_keras_tbl %>% recall(truth, estimate)
# F1-Statistic
estimates_keras_tbl %>% f_meas(truth, estimate, beta = 1)

####################################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

save(list = ls(), file = 'multiclassfication_malariaFinal.RData')


##The end

