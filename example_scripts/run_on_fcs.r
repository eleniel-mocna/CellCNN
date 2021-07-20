PATH_TO_PYTHON <- "c:/Users/souku/Documents/Motol/CellCNN/git/CellCNN/C/Scripts"
PATH_TO_FOLDER <- "T/"
SUMMARY_FILE <- "summary_c.csv"
SAMPLEID <- "SAMPLEID"
library(reticulate)

use_python(PATH_TO_PYTHON, T)
CellCNN <- import("CellCNN")
Models <- CellCNN$Models
D <- CellCNN$Dataset
library(flowCore)
summ <- read.csv(paste(PATH_TO_FOLDER, SUMMARY_FILE, sep=""))
summ[is.na(summ)]<-1
datasets <- list()
labels <- list()
fs <- read.flowSet(path = PATH_TO_FOLDER, pattern = "*.fcs")
i <- 1
for (code in summ[["Kod"]])
  {
  for (variable in vector)
    {
    if (j>length(fs)) {break}
    if (keyword(fs[[j]])[[SAMPLEID]]==code)
      {
      FlowFrame <- fs[[j]]
      FFData <- exprs(FlowFrame)
      datasets[[i]] <- D$DataDataset(FFData, shuffle=F)
      labels[[i]] <- summ[[j]]
      # If this crashes, see: #https://github.com/rstudio/reticulate/issues/831
      # Or just update reticulate with `remotes::install_github("rstudio/reticulate")`
      }
    j=j+1
  }
  i=i+1
}
         