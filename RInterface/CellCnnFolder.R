#' @export
CellCnnFolder <- R6::R6Class(
  "CellCnnFolder",
  inherit = CellCnnData,
  public = list(
    #' Create a new CellCnnData-like object from given path
    #' @param path path to the analysis folder (see README)
    #' @param NAME name for results of this analysis
    #' @param get_data_function function replacing `get_data`, e.g. transformations
    initialize = function(path,
                          NAME = NULL,
                          get_dato_function = NULL) {
      if (!is.null(get_dato_function)) {
        get_dato <- get_dato_function
      }
      path = normalizePath(path)
      private$.channels <-
        read.table(paste0(path, "/channels.tsv"), sep= "\t", header = TRUE)
      super$initialize(
        private$get_data(path),
        private$get_labels(path),
        private$get_label_descr(path),
        path,
        NAME
      )
    },
    #' For given fcs object return data that it contains.
    get_dato = function(fcs_object,
                        compensate = NULL,
                        fun = function(x)
                          asinh(x/120)) {
      # TODO: Does this work?
      if (!is.null(compensate)) {
        fcs_object <-
          flowCore::compensate(fcs_object, spillover = fcs_object@description[[compensate]])
      }
      return(fun(flowCore::exprs(fcs_object)[, private$.channels[(private$.channels[["use"]] ==
                                                                    1), "name"]]))
    }
  ),
  private = list(
    #' For given path, load all data and return it as a list of matrices
    #' sorted by the order of names in labels.tsv.
    #' @param path path to the analysis folder containing labels.tsv and
    #' data folder
    get_data = function(path) {
      names <-
        paste0(path, "/data/", rownames(private$get_labels(path)), ".fcs")
      fcss <- lapply(names, flowCore::read.FCS)
      return(lapply(fcss, self$get_dato))
    },
    
    #' For given path get labels as a matrix
    #' @param path path to the analysis folder containing labels.tsv
    get_labels = function(path,
                          which_file = "labels.tsv") {
      return(read.table(
        paste0(path, "/", which_file),
        sep = "\t",
        header = TRUE,
        row.names = 1
      ))
    },
    get_label_descr = function(path) {
      return(read.table(
        paste0(path, "/label_description.tsv"),
        sep = "\t",
        header = TRUE,
        row.names = 1
      ))
    },
    .channels = matrix()
  )
)