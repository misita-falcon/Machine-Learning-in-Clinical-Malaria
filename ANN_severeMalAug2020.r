#Script for developing binary model for SM and nMI
#######################################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#load neccessary packages, not really important but not neccessary
library(pacman)
pacman::p_load(ggplot2, reshape2, gplots, grid, spatstat, raster, sp, dplyr, 
               klaR, ggfortify, stringr, cluster, Rtsne, readr, RColorBrewer, Hmisc, mice, tidyr, 
               purrr, VIM, magrittr, corrplot, caret, gridExtra, ape, tidytree, pheatmap, stats, 
               vegan, FactoMineR, factoextra, outliers, ggpubr, keras, lime, tidyquant, rsample, 
               recipes, corrr, yardstick, tensorflow, caret, limma, compareGroups, forcats)
#S####################################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
setwd("/home/root01/Documents/machine_learn_sep_2020/Analysis/")

#get pre-processed data
clinhem <- read.csv("../Data/Imputed_sm_nmi.csv", header = T, na.strings = T); glimpse(clinhem)
#######################################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#remove the X column, hb_level, hematocrit
clinhem <- clinhem %>% select(-c(X, hb_level, hematocrit,location, patient_age)); glimpse(clinhem)
#randomize the data 
clinhem <- clinhem[sample(1:nrow(clinhem)), ]
# Split test/training sets
set.seed(1000)
train_test_split <- initial_split(clinhem, prop = 0.8); train_test_split
## Retrieve train and test sets
train_tbl_with_ids <- training(train_test_split); test_tbl_with_ids  <- testing(train_test_split)
train_tbl <- select(train_tbl_with_ids, -SampleID) ; test_tbl <- select(test_tbl_with_ids, -SampleID)
# Create recipe
recipe_SM <- recipe(Clinical_Diagnosis ~ ., data = train_tbl) %>%
  #step_dummy(all_nominal(), -all_outcomes()) %>%
  step_YeoJohnson(all_predictors(), -all_outcomes()) %>%
  step_center(all_predictors(), -all_outcomes()) %>%
  step_scale(all_predictors(), -all_outcomes()) %>%
  prep(data = train_tbl)
recipe_SM
# Predictors
x_train_tbl2 <- bake(recipe_SM, new_data = train_tbl); x_test_tbl2  <- bake(recipe_SM, new_data = test_tbl)
x_train_tbl <- x_train_tbl2 %>% select(-Clinical_Diagnosis) ; x_test_tbl <- x_test_tbl2 %>% select(-Clinical_Diagnosis)
# Response variables for training and testing sets
y_train_vec <- ifelse(pull(train_tbl, Clinical_Diagnosis) == "Severe Malaria", 1, 0)
y_test_vec  <- ifelse(pull(test_tbl, Clinical_Diagnosis) == "Severe Malaria", 1, 0)

######################################################################################################################

# Building our Artificial Neural Network
model_keras <- keras_model_sequential()
model_keras %>%
  # First hidden layer and Dropout to prevent overfitting
  layer_dense(units = 16, kernel_initializer = "uniform", activation = "relu", input_shape = ncol(x_train_tbl),
              kernel_regularizer = regularizer_l2(0.001)) %>% layer_dropout(rate = 0.2) %>% layer_batch_normalization() %>%
  # Second hidden layer and Dropout to prevent overfitting
  layer_dense(units = 128, kernel_initializer = "uniform", activation= "relu", 
              kernel_regularizer = regularizer_l2(0.001)) %>% layer_dropout(rate = 0.4) %>% layer_batch_normalization() %>%
  # Third hidden layer and Dropout to prevent overfitting
  layer_dense(units = 256, kernel_initializer = "uniform", activation= "relu", 
              kernel_regularizer = regularizer_l2(0.001)) %>% layer_dropout(rate = 0.1) %>% layer_batch_normalization() %>%
  # Output layer
  layer_dense(units= 1, kernel_initializer = "uniform",
              activation = "sigmoid") %>%
  # Compile ANN and backpropagation
  compile(
    optimizer = 'adam',
    loss      = 'binary_crossentropy',
    metrics   = c('accuracy')) ; model_keras
# Fit the keras model to the training data
fit_keras <- fit(
  object           = model_keras, 
  x                = as.matrix(x_train_tbl), 
  y                = y_train_vec,
  batch_size       = 64, 
  epochs           = 500,
  validation_split = 0.30, #for cross validation
  shuffle          = TRUE,
  verbose  = FALSE,
  callbacks = list(
    #callback_early_stopping(patience = 50),
    callback_tensorboard("run_Severe"),
    callback_reduce_lr_on_plateau(factor = 0.001)
  )
) ; #tensorboard("run_Severe") ; 
fit_keras # Print the final model

save_model_hdf5(model_keras, 'nMI_SM_malaria_Final.hdf5')
# Plot the training/validation history of our Keras model
plot_keras <- plot(fit_keras) +
  theme_tq() +
  scale_color_tq() +
  scale_fill_tq() 
#labs(title = "Accuracy and loss of during Training for Severe malaria") ;
plot_keras

#######################################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Predicted Class
yhat_keras_class_vec <- predict_classes(object = model_keras, x = as.matrix(x_test_tbl)) %>%
  as.vector()
# Predicted Class Probability
yhat_keras_prob_vec  <- predict_proba(object = model_keras, x = as.matrix(x_test_tbl)) %>%
  as.vector()
# Format test data and predictions for yardstick metrics
estimates_keras_tbl <- tibble(
  truth      = as.factor(y_test_vec) %>% fct_recode(Severe = "1", nonMalaria = "0"),
  estimate   = as.factor(yhat_keras_class_vec) %>% fct_recode(Severe = "1", nonMalaria = "0"),
  class_prob = yhat_keras_prob_vec
) ; estimates_keras_tbl

options(yardstick.event_first = FALSE)

# Confusion Table
estimates_keras_tbl %>% conf_mat(truth, estimate)
# Accuracy
estimates_keras_tbl %>% metrics(truth, estimate)
# AUC
estimates_keras_tbl %>% roc_auc(truth, class_prob)
#######!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(pROC)
# Get the probality threshold for specificity = 0.6
pROC_obj <- roc(estimates_keras_tbl$truth, estimates_keras_tbl$class_prob,
                smoothed = TRUE,
                # arguments for ci
                ci=FALSE, ci.alpha=0.9, stratified=FALSE,
                # arguments for plot
                plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
                print.auc=TRUE, show.thres=TRUE,  print.thres = c(0.1, 0.5, 0.8))
##########@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Precision
estimates_keras_tbl %>% precision(truth, estimate)
estimates_keras_tbl %>% recall(truth, estimate)

# F1-Statistic
estimates_keras_tbl %>% f_meas(truth, estimate, beta = 1)
class(model_keras)

####################################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Setup lime::model_type() function for keras
model_type.keras.engine.sequential.Sequential <- function(x, ...) {
  return("classification")
}
# Setup lime::predict_model() function for keras
predict_model.keras.engine.sequential.Sequential <- function(x, newdata, type, ...) {
  pred <- predict_proba(object = x, x = as.matrix(newdata))
  return(data.frame(Severe = pred, nonMalaria = 1 - pred))
}
# Setup lime::model_type() function for keras
model_type.keras.models.Sequential <- function(x, ...) {
  return("classification")
}
predictions <- predict_model(x = model_keras, newdata = x_test_tbl, type = 'raw') %>%
  tibble::as_tibble()
test_tbl_with_ids$churn_prob <- predictions$Severe
# Run lime() on training set
explainer <- lime::lime(
  x              = x_train_tbl, 
  model          = model_keras, 
  bin_continuous = FALSE)
# Run explain() on explainer
explanation <- lime::explain(
  x_test_tbl[76:79,],
  explainer    = explainer, 
  n_labels     = 1, 
  n_features   = 16,
  kernel_width = 0.5)
Featurebars <- plot_features(explanation) +
  labs(title = "Compact visual representation of feature importance in  cases",
       subtitle = "Severe Malaria compared to Non-malaria infections")
Featurebars
# Run explain() on explainer
explanation2 <- lime::explain(
  x_test_tbl[1:300,],
  explainer    = explainer, 
  n_labels     = 1, 
  n_features   = 15,
  kernel_width = 0.5)
library(plyr)
#plot heatmap
x <-  explanation2$feature ; y <- explanation2$feature_weight ; z <- explanation2$label ;w <- explanation2$case
x_name <- "feature" ; y_name <- "feature_weight" ;z_name <- "Disease Outcome"; w_name <- "case"
df <- data.frame(w,z,x,y); names(df) <- c(w_name, z_name, x_name,y_name) ; glimpse(df)
table(df_wide_2$`Disease Outcome`)

df$`Disease Outcome` <- revalue(df$`Disease Outcome`, c("Severe"="Severe Malaria", "nonMalaria"="Non-malaria Infections"))
df_wide <- spread(df, key = feature, value = feature_weight); df_wide <- df_wide[order(df_wide$`Disease Outcome`),]
table(df_wide$`Disease Outcome`) ; df_wideA <- slice(df_wide, 1:300); dim(df_wideA)

df_wide_2 <- df_wideA[, -2] ; row.names(df_wide_2) <- df_wide_2$case ; df_wide_2[1] <- NULL
df_Wide_dem <- df_wideA[,-(3:17)]; row.names(df_Wide_dem) <- df_Wide_dem$case
df_Wide_dem[1] <- NULL; df_matx <- as.matrix(df_wide_2)

inde <- read.csv("../Data/indices.csv", row.names = 1)

mat_colors <- list(group = brewer.pal(3, "Set2"))
  names(mat_colors$group) <- unique(inde$Index.type)


pheatmap(df_matx, annotation_col = inde, cutree_rows = 2, clustering_distance_cols = "correlation", 
         cluster_rows = TRUE, cluster_cols = TRUE, annotation_row = df_Wide_dem, annotation_colors = mat_colors,  fontsize = 12, show_rownames = F)
####################################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

save(list = ls(), file = 'nMI_SM_malariaFinal.RData')
