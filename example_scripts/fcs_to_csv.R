# This script takes fcs files from directory "./T" and creates csv files from them (in dir "./csv")
# And copies file "T/tabulka.csv" into "./summary.csv"

library(flowCore)
library(jsonlite)

SUMMARY_NAME <- "summary"
ID_KEYWORD <- "SAMPLEID"
CSV_DIRECTORY <- "csv"
FCS_DIRECTORY <- "T"
SEP <- "/"

dir.create(CSV_DIRECTORY)
save_csv <- function(fcs_data)
{
  # Saves given flowFrame object to csv and json
  meta_data <- keyword(fcs_data)
  id <- meta_data[[ID_KEYWORD]]
  data_path <- paste(CSV_DIRECTORY, SEP, id, ".csv", sep = "")
  raw_data <- exprs(fcs_data)
  write.csv(raw_data, data_path)
  
  meta_path <- paste(CSV_DIRECTORY, SEP, id, ".json", sep = "")
  exportJSON <- toJSON(meta_data)
  write(exportJSON, meta_path)
}

process_fcs <- function(file_name)
{
  write(paste("Processing file", file_name, "..."), stdout())
  fcs_data <-  read.FCS(file_name)
  # Here any manipulation with data can be done
  save_csv(fcs_data)
  write(paste(file_name, "Proccessed Successfully!"), stdout())
}

write(paste("Translating files from", FCS_DIRECTORY, "..."), stdout())
files <- list.files(path=FCS_DIRECTORY, pattern="*.fcs", full.names=TRUE, recursive=FALSE)
lapply(files, process_fcs)

write("Copying summary file...", stdout())
summaries <- list.files(path=FCS_DIRECTORY, pattern="*.csv", full.names=TRUE, recursive=FALSE)
file.copy(from = summaries[[1]],
          to=paste(CSV_DIRECTORY, SEP, SUMMARY_NAME, ".csv", sep=""))
write("Files translated successfully!", stdout())

