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
                          path=NULL,
                          name=NULL){
      private$.data <- data
      private$.labels <- labels
      private$.label_description <- label_descr
      private$set_path(path)
      private$set_name(name)
    },
    
    #' Validate if given data is correct
    validate = function() {
      # Check number of samples:
      stopifnot(self$n_samples==length(private$.data))
      stopifnot(dim(private$.labels)[1]==length(private$.data))
      
      # Check data integrity
      for (dato in private$.data) {
        stopifnot(dim(dato)[2]==self$n_channels)
      }
      
      # Check label descriptions
      stopifnot(all(c("name", "type") %in% colnames(private$.label_description)))
      
    
      self$validate_labels()
      
      return(invisible(self))
    },
    
    #' Print basic info about this object.
    print = function() {
      cat("<CellCnnData>:", private$.name, "@", private$.path, "\n")
      cat("- n_samples: ", self$n_samples, "\n")
      cat("- n_channels:", self$n_channels, "\n")
      cat("- labels:", paste(colnames(private$.labels), collapse = ", "), "\n")
      return(invisible(self))
    },
    
    #' Set all <NA> values to -1
    clean_labels = function(){
      private$.labels[is.na(private$.labels)] <- -1
    },
    validate_labels = function(){
      stopifnot(!unlist(lapply(private$.labels, is.character)))
    }
  ),
  
  active = list(
    #' Number of samples in this dataset
    n_samples = function(value) {
      if (missing(value)){
        return(length(private$.data))
      }
      else {
        stop("Cannot overwrite `n_samples` attribute!")
      }
    },
    
    #' Number of channels in this dataset
    n_channels = function(value) {
      if (missing(value)){
        return(dim(private$.data[[1]])[2])
      }
      else {
        stop("Cannot overwrite `n_channels` attribute!")
      }
    },
    
    #' Getter for data in this dataset
    data = function(value) {
      if (missing(value)){
        return(private$.data)
      }
      else {
        stop("Cannot overwrite data! Create a new object instead!")
      }
    },
    
    #' Getter for labels in this dataset
    labels = function(value) {
      if (missing(value)){
        return(private$.labels)
      }
      else {
        stop("Cannot overwrite labels! Create a new object instead!")
      }
    },
    
    #' Getter for label description in this dataset
    label_description = function(value) {
      if (missing(value)){
        return(private$.label_description)
      }
      else {
        stop("Cannot overwrite labels! Create a new object instead!")
      }
    }
  ),
  
  private = list(
    .data = list(),
    .labels = matrix(),
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
