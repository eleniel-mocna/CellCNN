library(reticulate)
use_condaenv("/home/rstudio/.local/share/r-miniconda/envs/r-reticulate/bin/python")
library(flowCore)
#devtools::install_github("https://github.com/cipheLab/FlowCIPHE.git")
library(FlowCIPHE)
CellCNN <- import("CellCNN")
rscripts <- import("CellCNN.rscripts")
np <- import("numpy")

get_values <- function(fcs) {
  ret <- exprs(fcs)[,channels[(channels[["use"]]==1),"name"]]
  asinh(ret/5)
}

channels <- read.table("BCa_channels.txt", header = TRUE)

data_table <- read.table("BCa_cohort_souceklab.txt", sep="\t", header=TRUE, na.strings = "x")
file_names <- paste(data_table[["X"]], "_vaevictis.fcs", sep="")
fcs_files <- lapply(file_names, read.FCS)
data <- lapply(fcs_files, get_values)

labels <- data_table[,c("pT", "pT", "pN", "ki.67", "neoadj..CHT", "age")]
labels[,1][labels[,1]=="1c"] = 1
labels[,1] <- strtoi(labels[,1]) - 1
labels[,2] <- ifelse(labels[,2]=="1c",yes = 1, no = 0)

class_description = c(3L, 2L, 2L, 0L, 2L, 0L)
model <- rscripts$train_model(data = data,
                              labels = labels,
                              multicell_size = 1000L,
                              amount = 100000L,
                              test_amount = 20000L,
                              layers = matrix(16L),
                              epochs = 10L,
                              classes = class_description)

sm <- model$get_single_cell_model()
results <- lapply(data, sm)
results <- lapply(results, np$array)
for (k in 1:length(fcs_files)){
  a <- results[[k]][,8]
  a <- ifelse(a==0, yes=0, no=1)
  b <- results[[k]][,10]
  b <- ifelse(b==0, yes=0, no=1)
  color <- cbind(a, b, results[[k]][,9], results[[k]][,9])
  d <- exprs(fcs_files[[k]])
  scattermoreplot(d[,"vaevictis_1"],
                 d[,"vaevictis_2"],
                 col=rgb(color))
}
for (k in 1:length(fcs_files)) {print(paste("Processing file", k))
  for (i in c(8,10)){
    fcs_files[[k]] <- enrich.FCS.CIPHE(fcs_files[[k]],
                                       results[[k]][,i],
                                       nw.names = list(paste("Filter", i)))
  }
  write.FCS(fcs_files[[k]], gsub("_vaevictis.fcs", "_enriched_vaevictis.fcs", file_names[[k]]))
}

# This helps print the right filters ([n] -> filter n + 1)
for (i in 0:5) {
  print(model$layers[[6+i]]$weights[[1]][7]$numpy())
  print(model$layers[[6+i]]$weights[[1]][9]$numpy())
  print(paste(i, "----------------"))
}

