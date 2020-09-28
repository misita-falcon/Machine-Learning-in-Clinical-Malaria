library(pacman)
library(remotes)
#Helper packages
library(dplyr)         # for basic data wrangling
# Modeling packages
library(keras)         # for fitting DNNs
library(tfruns)        # for additional grid search & model training functions
# Modeling helper package - not necessary for reproducibility
library(tfestimators)  # provides grid search & model training interface

pythonpath='/opt/apps/Python/Python-3.6.8/bin/python3'
Sys.setenv(RETICULATE_PYTHON=pythonpath)
library(reticulate)
library(keras)
library(tensorflow)
#confirm
reticulate::py_discover_config()

# Run various combinations of dropout1 and dropout2
runs <- tuning_run("/home/cmmoranga/machine_learn/um/um_nmI_tuning.R", 
                   flags = list(
                     nodes1 = c(16, 32, 64, 128, 256, 512),
                     nodes2 = c(16, 32, 64, 128, 256, 512),
                     nodes3 = c(16, 32, 64, 128, 256, 512),
                     dropout1 = c(0.1, 0.2, 0.3, 0.4),
                     dropout2 = c(0.1, 0.2, 0.3, 0.4),
                     dropout3 = c(0.1, 0.2, 0.3, 0.4),
                     optimizer = c("adam"),
                     lr_annealing = c(0.1, 0.05, 0.001, 0.0001)
                   ),
                   sample = 0.10
)

UM_nMI <- runs %>% 
  filter(metric_val_loss == min(metric_val_loss)) %>% 
  glimpse()

write.csv(UM_nMI, "/home/cmmoranga/machine_learn/um/um_nmi.csv")
