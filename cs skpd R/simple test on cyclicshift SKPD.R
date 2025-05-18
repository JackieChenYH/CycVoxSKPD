source("skpd_tensor_utils.R")
source("skpd_classifier.R")

set.seed(1)
hp <- list(p1=2,d1=2,p2=2,d2=2,p3=1,d3=1,
           term=2,lmbda_a=0,lmbda_b=0,lmbda_gamma=0,
           alpha=0.25,normalization_A=FALSE,normalization_B=FALSE,
           use_cyclic=TRUE)

make_sample <- function(y){
  Ctrue <- matrix(c(2,0,0,0,
                    0,2,0,0,
                    0,0,0,0,
                    0,0,0,0),4,4)
  X <- matrix(rnorm(16),4,4) + y*Ctrue
  list(X, X, y)
}
Ds <- lapply(sample(0:1,20,TRUE), make_sample)

mod <- SKPD_LogisticRegressor_Cyclic$new(hp, Ds=Ds, max_iter=5, verbose=TRUE)
mod$fit()
