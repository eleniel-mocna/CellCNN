library(reticulate)
use_condaenv("/home/rstudio/.local/share/r-miniconda/envs/r-reticulate/bin/python")
library(flowCore)
library(FlowCIPHE)

tabulka_path <- "tabulka.csv"
CellCNN <- import("CellCNN")
rscripts <- import("CellCNN.rscripts")
np <- import("numpy")
tabulka <- read.table(tabulka_path, header = TRUE)

get_values <- function(fcs) {
  fcs@exprs[,12:20]
}

first_names <- list("ZIK 7353_27_8_28_38__CD.fcs", "ZIB7001_27_8_28_38_5_CD.fcs", "WOC 6652_27_8_28_38__CD.fcs")
second_names <- list("THON_27_8_28_38_57_3_CD.fcs", "UNLK1_27_8_28_38_57__CD.fcs", "TRAVNICKOVA_27_8_28__CD.fcs")
first_fcs <- lapply(first_names, read.FCS)
second_fcs <- lapply(second_names, read.FCS)


first_values <- lapply(first_fcs, get_values)
second_values <- lapply(second_fcs, get_values)
labels <- matrix(c(1,1,1,0,0,0),ncol = 1)

data <- c(first_values, second_values)

model <- rscripts$train_model(data = data,
                              labels = labels,
                              multicell_size = 1000L,
                              amount = 1000L,
                              test_amount = 200L,
                              layers = c(64L,64L, 4L),
                              epochs = 10L)

sm <- model$get_single_cell_model()
first_results <- lapply(first_values, sm)
first_results <- lapply(first_results, np$array)
second_results <- lapply(second_values, sm)
second_results <- lapply(second_results, np$array)

for (i in 1:length(second_results[[1]][1,])){
  first_fcs[[1]] <- enrich.FCS.CIPHE(first_fcs[[1]],
                                     first_results[[1]][,i],
                                     nw.names = list(paste("Filter", i, ":",
                                                           model$layers[[5]]$weights[[1]][i-1]$numpy())))
  second_fcs[[1]] <- enrich.FCS.CIPHE(second_fcs[[1]],
                                      second_results[[1]][,i],
                                      nw.names = list(paste("Filter", i, ":",
                                                            model$layers[[5]]$weights[[1]][i-1]$numpy())))
}
write.FCS(first_fcs[[1]],paste("W_FILTERS_", first_names[[1]]))
write.FCS(second_fcs[[1]],paste("W_FILTERS_", second_names[[1]]))
