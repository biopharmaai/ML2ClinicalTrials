# filepath: [function.R](http://_vscodecontentref_/0)

# Require the Python trialbench package
.onLoad <- function(libname, pkgname) {
  # Check if the trialbench package is installed
  if (!reticulate::py_module_available("trialbench")) {
    stop("The trialbench Python package is not installed. Please install it using 'reticulate::py_require(c(\"trialbench\"))'.")
  }
  # Import the installed trialbench module
  trialbench <- reticulate::import("trialbench")
  # Assign the module object to the package environment
  assign("trialbench", trialbench, envir = parent.env(environment()))
}


#' Download all datasets
#'
#' @param save_path String, path to save the data
#' @export
download_all_data <- function(save_path) {
  tryCatch({
    trialbench <- reticulate::import("trialbench")
    trialbench$download_all_data(save_path)
  }, error = function(e) {
    stop("Error occurs: ", e$message)
  })
}

#' Load clinical trial data
#'
#' Load clinical trial data for specified task and phase by calling Python's trialbench.load_data function
#'
#' @param task String, dataset name, such as 'mortality_rate'
#' @param phase String, clinical trial phase, such as 'Phase 1', 'Phase 2', 'Phase 3', 'Phase 4'
#' @return List, containing train_df, valid_df, test_df, num_classes and tabular_input_dim
#' @export
load_data <- function(task, phase) {
  tryCatch({
    trialbench <- reticulate::import("trialbench")
    
    result <- trialbench$load_data(task, phase)
    return(list(
      train_df = as.data.frame(reticulate::py_to_r(result[[1]])),
      valid_df = as.data.frame(reticulate::py_to_r(result[[2]])),
      test_df = as.data.frame(reticulate::py_to_r(result[[3]])),
      num_classes = reticulate::py_to_r(result[[4]]),
      tabular_input_dim = reticulate::py_to_r(result[[5]])
    ))
  }, error = function(e) {
    stop("Error occurs: ", e$message)
  })
}