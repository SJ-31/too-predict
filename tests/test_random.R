library(reticulate)
library(ALDEx2)
rdirichlet <- function(n, a)
                       ## pick n random deviates from the Dirichlet function with shape parameters a
## a are the concentration parameters
{
  if (length(n) > 1 || length(n) < 1 || n < 1) stop("n must be a single positive integer value")
  if (length(a) < 2) stop("a must be a vector of numeric value")
  n <- floor(n)
  l <- length(a)
  x <- matrix(rgamma(l * n, a), ncol = l, byrow = TRUE)
  return(x / rowSums(x))
}

# Calculate dirichlet from (len(a) * n) random samples from gamma distribution
# (variant with shape and scale parameters),
# with shape a

stats <- import("scipy.stats")

data(selex)
test <- selex[300:700, ]

lapply(test, \(x) print(x))

rdirichlet(10, test[[1]])

rd <- rdirichlet(1, test[[1]])

scipy <- stats$dirichlet$rvs(as.integer(test[[1]]), size = as.integer(1))
