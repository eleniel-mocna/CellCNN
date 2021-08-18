###### Setup ######

library(reticulate)
use_condaenv("/home/rstudio/.local/share/r-miniconda/envs/r-reticulate/bin/python")
CellCNN <- import("CellCNN")
library(flowCore)
#devtools::install_github("https://github.com/cipheLab/FlowCIPHE.git")
library(FlowCIPHE)
library(scattermore)
rscripts <- import("CellCNN.rscripts")
np <- import("numpy")

###### Data ######

get_values <- function(fcs) {
  ret <- exprs(fcs)[,channels[(channels[["use"]]==1),"name"]]
  asinh(ret/5)
}

channels <- read.table("BCa_channels.txt", header = TRUE)

data_table <- read.table("BCa_cohort_souceklab.txt", sep="\t", header=TRUE, na.strings = "x")
file_names <- paste(data_table[["X"]], "_vaevictis.fcs", sep="")
fcs_files <- lapply(file_names, read.FCS)
data <- lapply(fcs_files, get_values)

###### Labels #######

labels <- data_table[,c("pT", "pT", "pN", "ki.67", "neoadj..CHT", "age")]
labels[,1][labels[,1]=="1c"] = 1
labels[,1] <- strtoi(labels[,1]) - 1
labels[,2] <- ifelse(labels[,2]=="1c",yes = 1, no = 0)

class_description = c(3L, 2L, 2L, 0L, 2L, 0L)

###### Training model and getting results ######

model <- rscripts$train_model(data = data,
                              labels = labels,
                              multicell_size = 1000L,
                              amount = 10000L,
                              test_amount = 2000L,
                              layers = list(16L),
                              epochs = 2L,
                              classes = class_description)

sm <- model$get_single_cell_model()
results <- lapply(data, sm)
results <- lapply(results, np$array)

###### Visualisation ######

for (k in 1:length(fcs_files)){
  # 2,3,5,6,7,8,9,11,12,13,14,15
  i <- 13
  j <- 14
  l <- 15
  a <- results[[k]][,i]
  a <- ifelse(a==0, yes=0, no=0.8)
  b <- results[[k]][,j]
  b <- ifelse(b==0, yes=0, no=0.8)
  c <- results[[k]][,l]
  c <- ifelse(c==0, yes=0, no=0.8)
  color <- cbind(a, b, c, results[[k]][,10])
  d <- exprs(fcs_files[[k]])
  scattermoreplot(d[,"vaevictis_1"],
                 d[,"vaevictis_2"],
                 col=rgb(color),
                 cex = 1.3)
}

###### Saving files ######

for (k in 1:length(fcs_files)) {print(paste("Processing file", k))
  for (i in c(2,3,5,6,7,8,9,11,12,13,14,15)){
    fcs_files[[k]] <- enrich.FCS.CIPHE(fcs_files[[k]],
                                       results[[k]][,i],
                                       nw.names = list(paste("Filter", i)))
  }
  write.FCS(fcs_files[[k]], gsub("vaevictis.fcs", "enriched_single2.fcs", file_names[[k]]))
}

###### Getting filter weights  ######

# This helps print the right filters ([n] -> filter n + 1)
f <- file("output.txt", open="wt")
first_output_layer <- 4
for (i in 0:5) {
  for (k in c(2,3,5,6,7,8,9,11,12,13,14,15)){
    writeLines(paste("\n Output", i, " Filter: ", k, "\t"), f)
    writeLines(paste("", model$layers[[first_output_layer+i]]$weights[[1]][k-1]$numpy()), f)
    
  }
}
close(f)
