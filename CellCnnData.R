CellCnnData <- R6::R6Class(
  "CellCnnData",
  public = list(
    #' Initialize an object containing data for analysis.
    #' @param data: list of matrices -> a matrix for every sample,
    #'     containing (n_cells, data dimensions)
    #' @param labels: matrix (n_samples, n_labels)
    #' @param label_descr: data frame (n_labels, (...,"type"))
    #'     dataframe, where rownames are label names and
    #'     which contains a column named "type".
    #' @param path: String leading to a folder, where analyses will be stored
    #'     (getwd() by default)
    #' @param name: Name of this data set.
    initialize = function(data,
                          labels,
                          label_descr,
                          path = NULL,
                          name = NULL) {
      private$.data <- data
      self$labels <- labels
      private$.label_description <- label_descr
      private$set_path(path)
      private$set_name(name)
    },
    
    #' Validate if given data is correct
    validate = function() {
      # Check number of samples:
      stopifnot(self$n_samples == length(private$.data))
      stopifnot(dim(self$labels)[1] == length(private$.data))
      
      # Check data integrity
      for (dato in private$.data) {
        stopifnot(dim(dato)[2] == self$n_channels)
      }
      
      # Check label descriptions
      stopifnot(all(c("name", "type") %in% colnames(private$.label_description)))
      
      
      self$validate_labels()
      
      return(invisible(self))
    },
    
    #' Print basic info about this object.
    print = function() {
      cat("<",
          class(self)[1],
          "> :",
          private$.name,
          "@",
          private$.path,
          "\n")
      cat("- n_samples: ", self$n_samples, "\n")
      cat("- n_channels:", self$n_channels, "\n")
      cat("- labels:", paste(colnames(self$labels), collapse = ", "), "\n")
      return(invisible(self))
    },
    
    #' Set all <NA> values to -1
    clean_labels = function() {
      self$labels[self$labels == -1] <- NA
      normal_data <- function(x)
      {
        x <- (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
        x - min(x, na.rm = TRUE)
      }
      for (i in 1:nrow(private$.label_description)) {
        if ((private$.label_description)[i, "type"] == 0
            &&
            rownames(private$.label_description)[i] %in% colnames(self$labels)) {
          self$labels[, rownames(private$.label_description)[i]] = (normal_data(self$labels[, rownames(private$.label_description)[i]]))
        }
      }
      self$labels[is.na(self$labels)] <- -1
      private$.label_description <- (private$.label_description[rownames(private$.label_description)
                                                                %in% colnames(self$labels), , drop =
                                                                  FALSE])
      self$labels <- self$labels[, colnames(self$labels)
                                 %in% rownames(private$.label_description), drop =
                                   FALSE]
      
      
    },
    validate_labels = function() {
      stopifnot(!unlist(lapply(self$labels, is.character)))
    },
    labels = matrix()
  ),
  
  active = list(
    #' Number of samples in this dataset
    n_samples = function(value) {
      if (missing(value)) {
        return(length(private$.data))
      }
      else {
        stop("Cannot overwrite `n_samples` attribute!")
      }
    },
    
    #' Path to the analysis folder
    path = function(value) {
      if (missing(value)) {
        return(private$.path)
      }
      else {
        stop("Cannot overwrite `path` attribute!")
      }
    },
    
    #' Name of this analysis results (path+name = results folder)
    name = function(value) {
      if (missing(value)) {
        return(private$.name)
      }
      else {
        stop("Cannot overwrite `name` attribute!")
      }
    },
    
    #' Number of channels in this dataset
    n_channels = function(value) {
      if (missing(value)) {
        return(dim(private$.data[[1]])[2])
      }
      else {
        stop("Cannot overwrite `n_channels` attribute!")
      }
    },
    
    #' Getter for data in this dataset
    data = function(value) {
      if (missing(value)) {
        return(private$.data)
      }
      else {
        stop("Cannot overwrite data! Create a new object instead!")
      }
    },
    
    #' Getter for label description in this dataset
    label_description = function(value) {
      if (missing(value)) {
        return(private$.label_description)
      }
      else {
        stop("Cannot overwrite labels! Create a new object instead!")
      }
    }
  ),
  
  private = list(
    .data = list(),
    .label_description = matrix(),
    .path = "",
    .name = "",
    
    set_path = function(path) {
      if (is.null(path)) {
        private$.path <- getwd()
      }
      else {
        stopifnot(is.character(path))
        private$.path <- normalizePath(path)
      }
    },
    
    set_name = function(name) {
      if (is.null(name)) {
        private$.name <- private$.path
      }
      else {
        stopifnot(is.character(name))
        private$.name <- name
      }
    }
  )
)
