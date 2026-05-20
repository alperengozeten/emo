# Robust Regression Implementation
# This baseline fits ordinary least squares with an intercept.

fit_model <- function(X_train, y_train) {
  # EVOLVE-BLOCK-START
  X_matrix <- as.matrix(X_train)
  y_vector <- as.numeric(y_train)
  design_matrix <- cbind(1.0, X_matrix)

  qr_fit <- qr(design_matrix)
  coefficients <- qr.coef(qr_fit, y_vector)
  coefficients <- as.numeric(coefficients)
  coefficients[!is.finite(coefficients)] <- 0.0

  list(coefficients = coefficients)
  # EVOLVE-BLOCK-END
}


predict_model <- function(model, X_test) {
  X_matrix <- as.matrix(X_test)
  design_matrix <- cbind(1.0, X_matrix)
  as.numeric(design_matrix %*% as.numeric(model$coefficients))
}


main <- function(X_train, y_train, X_test) {
  model <- fit_model(X_train, y_train)
  predictions <- predict_model(model, X_test)
  list(
    coefficients = as.numeric(model$coefficients),
    predictions = as.numeric(predictions)
  )
}
