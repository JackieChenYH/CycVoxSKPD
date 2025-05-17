# skpd_utils.R

library(R6)

# Ten2MatOperator
Ten2MatOperator <- R6Class("Ten2MatOperator",
                           public = list(
                             p1 = NULL, d1 = NULL, p2 = NULL, d2 = NULL, p3 = NULL, d3 = NULL,
                             initialize = function(p1, d1, p2, d2, p3 = 1, d3 = 1) {
                               self$p1 <- p1
                               self$d1 <- d1
                               self$p2 <- p2
                               self$d2 <- d2
                               self$p3 <- p3
                               self$d3 <- d3
                             },
                             forward = function(C) {
                               if (length(dim(C)) == 2) {
                                 m <- nrow(C)
                                 n <- ncol(C)
                                 stopifnot(m == self$p1 * self$d1, n == self$p2 * self$d2)
                                 RC <- list()
                                 for (i in 1:self$p1) {
                                   for (j in 1:self$p2) {
                                     Cij <- C[(self$d1 * (i-1) + 1):(self$d1 * i),
                                              (self$d2 * (j-1) + 1):(self$d2 * j)]
                                     RC[[length(RC) + 1]] <- as.vector(Cij)
                                   }
                                 }
                                 return(t(do.call(rbind, RC)))
                               } else if (length(dim(C)) == 3) {
                                 m <- dim(C)[1]
                                 n <- dim(C)[2]
                                 d <- dim(C)[3]
                                 stopifnot(m == self$p1 * self$d1, n == self$p2 * self$d2, d == self$p3 * self$d3)
                                 RC <- list()
                                 for (i in 1:self$p1) {
                                   for (j in 1:self$p2) {
                                     for (k in 1:self$p3) {
                                       Cij <- C[(self$d1 * (i-1) + 1):(self$d1 * i),
                                                (self$d2 * (j-1) + 1):(self$d2 * j),
                                                (self$d3 * (k-1) + 1):(self$d3 * k)]
                                       RC[[length(RC) + 1]] <- as.vector(Cij)
                                     }
                                   }
                                 }
                                 return(t(do.call(rbind, RC)))
                               } else {
                                 stop("Input must be a 2D or 3D array.")
                               }
                             }
                           )
)

# Mat2TenOperator
Mat2TenOperator <- R6Class("Mat2TenOperator",
                           public = list(
                             p1 = NULL, d1 = NULL, p2 = NULL, d2 = NULL, p3 = NULL, d3 = NULL,
                             initialize = function(p1, d1, p2, d2, p3 = 1, d3 = 1) {
                               self$p1 <- p1
                               self$d1 <- d1
                               self$p2 <- p2
                               self$d2 <- d2
                               self$p3 <- p3
                               self$d3 <- d3
                             },
                             forward = function(RC) {
                               if (self$p3 == 1 && self$d3 == 1) {
                                 p1p2 <- nrow(RC)
                                 d1d2 <- ncol(RC)
                                 C <- matrix(0, nrow = self$p1 * self$d1, ncol = self$p2 * self$d2)
                                 for (i in 1:p1p2) {
                                   Block <- matrix(RC[i, ], nrow = self$d1, ncol = self$d2)
                                   ith <- floor((i-1) / self$p2) + 1
                                   jth <- (i-1) %% self$p2 + 1
                                   C[(self$d1 * (ith-1) + 1):(self$d1 * ith),
                                     (self$d2 * (jth-1) + 1):(self$d2 * jth)] <- Block
                                 }
                                 return(C)
                               } else {
                                 p1p2p3 <- nrow(RC)
                                 d1d2d3 <- ncol(RC)
                                 C <- array(0, dim = c(self$p1 * self$d1, self$p2 * self$d2, self$p3 * self$d3))
                                 p2p3 <- self$p2 * self$p3
                                 for (i in 1:p1p2p3) {
                                   Block <- array(RC[i, ], dim = c(self$d1, self$d2, self$d3))
                                   ith <- floor((i-1) / p2p3) + 1
                                   reminder <- (i-1) %% p2p3
                                   jth <- floor(reminder / self$p3) + 1
                                   kth <- reminder %% self$p3 + 1
                                   C[(self$d1 * (ith-1) + 1):(self$d1 * ith),
                                     (self$d2 * (jth-1) + 1):(self$d2 * jth),
                                     (self$d3 * (kth-1) + 1):(self$d3 * kth)] <- Block
                                 }
                                 return(C)
                               }
                             }
                           )
)

# TranslationShiftOperator
TranslationShiftOperator <- R6Class("TranslationShiftOperator",
                                    public = list(
                                      shift_amount = NULL,
                                      initialize = function(shift_amount = c(1, 1, 1)) {
                                        self$shift_amount <- shift_amount
                                      },
                                      cs = function(tensor, forward = TRUE) {
                                        if (forward) {
                                          return(self$forward(tensor))
                                        } else {
                                          return(self$backward(tensor))
                                        }
                                      },
                                      forward = function(tensor) {
                                        if (length(dim(tensor)) == 2) {
                                          return(self$shift_2d(tensor, self$shift_amount[1:2]))
                                        } else if (length(dim(tensor)) == 3) {
                                          return(self$shift_3d(tensor, self$shift_amount))
                                        } else {
                                          stop("Input must be a 2D or 3D array.")
                                        }
                                      },
                                      backward = function(tensor) {
                                        if (length(dim(tensor)) == 2) {
                                          return(self$shift_2d(tensor, -self$shift_amount[1:2]))
                                        } else if (length(dim(tensor)) == 3) {
                                          return(self$shift_3d(tensor, -self$shift_amount))
                                        } else {
                                          stop("Input must be a 2D or 3D array.")
                                        }
                                      },
                                      shift_2d = function(tensor, shift) {
                                        s1 <- shift[1]
                                        s2 <- shift[2]
                                        p1 <- nrow(tensor)
                                        p2 <- ncol(tensor)
                                        shifted <- matrix(0, nrow = p1, ncol = p2)
                                        if (s1 >= 0 && s2 >= 0) {
                                          if (s1 < p1 && s2 < p2) {
                                            shifted[(s1+1):p1, (s2+1):p2] <- tensor[1:(p1-s1), 1:(p2-s2)]
                                          }
                                        } else if (s1 < 0 && s2 < 0) {
                                          shifted[1:(p1+s1), 1:(p2+s2)] <- tensor[(-s1+1):p1, (-s2+1):p2]
                                        }
                                        return(shifted)
                                      },
                                      shift_3d = function(tensor, shift) {
                                        s1 <- shift[1]
                                        s2 <- shift[2]
                                        s3 <- shift[3]
                                        p1 <- dim(tensor)[1]
                                        p2 <- dim(tensor)[2]
                                        p3 <- dim(tensor)[3]
                                        shifted <- array(0, dim = c(p1, p2, p3))
                                        if (s1 >= 0 && s2 >= 0 && s3 >= 0) {
                                          if (s1 < p1 && s2 < p2 && s3 < p3) {
                                            shifted[(s1+1):p1, (s2+1):p2, (s3+1):p3] <- tensor[1:(p1-s1), 1:(p2-s2), 1:(p3-s3)]
                                          }
                                        } else if (s1 < 0 && s2 < 0 && s3 < 0) {
                                          shifted[1:(p1+s1), 1:(p2+s2), 1:(p3+s3)] <- tensor[(-s1+1):p1, (-s2+1):p2, (-s3+1):p3]
                                        }
                                        return(shifted)
                                      }
                                    )
)

# CyclicShiftOperator
CyclicShiftOperator <- R6Class("CyclicShiftOperator",
                               public = list(
                                 cs = function(tensor, forward = TRUE, original_dims = NULL, scaling_factors = NULL) {
                                   if (length(dim(tensor)) == 2) {
                                     p1 <- nrow(tensor)
                                     p2 <- ncol(tensor)
                                     if (!is.null(original_dims) && !is.null(scaling_factors)) {
                                       orig_p1 <- original_dims[1]
                                       orig_p2 <- original_dims[2]
                                       d1 <- scaling_factors[1]
                                       d2 <- scaling_factors[2]
                                       mid_p1 <- floor(orig_p1 / 2) * d1
                                       mid_p2 <- floor(orig_p2 / 2) * d2
                                     } else {
                                       mid_p1 <- floor(p1 / 2)
                                       mid_p2 <- floor(p2 / 2)
                                     }
                                     idx1 <- if (forward) c((p1 - mid_p1 + 1):p1, 1:(p1 - mid_p1)) else c((mid_p1 + 1):p1, 1:mid_p1)
                                     idx2 <- if (forward) c((p2 - mid_p2 + 1):p2, 1:(p2 - mid_p2)) else c((mid_p2 + 1):p2, 1:mid_p2)
                                     shifted_tensor <- tensor[idx1, ]
                                     shifted_tensor <- shifted_tensor[, idx2]
                                   } else if (length(dim(tensor)) == 3) {
                                     p1 <- dim(tensor)[1]
                                     p2 <- dim(tensor)[2]
                                     p3 <- dim(tensor)[3]
                                     if (!is.null(original_dims) && !is.null(scaling_factors)) {
                                       orig_p1 <- original_dims[1]
                                       orig_p2 <- original_dims[2]
                                       orig_p3 <- original_dims[3]
                                       d1 <- scaling_factors[1]
                                       d2 <- scaling_factors[2]
                                       d3 <- scaling_factors[3]
                                       mid_p1 <- floor(orig_p1 / 2) * d1
                                       mid_p2 <- floor(orig_p2 / 2) * d2
                                       mid_p3 <- floor(orig_p3 / 2) * d3
                                     } else {
                                       mid_p1 <- floor(p1 / 2)
                                       mid_p2 <- floor(p2 / 2)
                                       mid_p3 <- floor(p3 / 2)
                                     }
                                     idx1 <- if (forward) c((p1 - mid_p1 + 1):p1, 1:(p1 - mid_p1)) else c((mid_p1 + 1):p1, 1:mid_p1)
                                     idx2 <- if (forward) c((p2 - mid_p2 + 1):p2, 1:(p2 - mid_p2)) else c((mid_p2 + 1):p2, 1:mid_p2)
                                     idx3 <- if (forward) c((p3 - mid_p3 + 1):p3, 1:(p3 - mid_p3)) else c((mid_p3 + 1):p3, 1:mid_p3)
                                     shifted_tensor <- tensor[idx1, , ]
                                     shifted_tensor <- shifted_tensor[, idx2, ]
                                     shifted_tensor <- shifted_tensor[, , idx3]
                                   } else {
                                     stop("Input must be a 2D or 3D array.")
                                   }
                                   return(shifted_tensor)
                                 }
                               )
)

# CS_SKPDDataSet
CS_SKPDDataSet <- R6Class("CS_SKPDDataSet",
                          public = list(
                            p1 = NULL, d1 = NULL, p2 = NULL, d2 = NULL, p3 = NULL, d3 = NULL,
                            use_cyclic = NULL, cyclic_shift = NULL,
                            X = NULL, X2 = NULL, Y = NULL, Z = NULL, n = NULL,
                            initialize = function(hparams, X, Y, Z = NULL) {
                              self$p1 <- hparams$p1
                              self$d1 <- hparams$d1
                              self$p2 <- hparams$p2
                              self$d2 <- hparams$d2
                              self$p3 <- hparams$p3
                              self$d3 <- hparams$d3
                              self$use_cyclic <- hparams$use_cyclic
                              self$cyclic_shift <- CyclicShiftOperator$new()
                              self$X <- X
                              self$Y <- Y
                              self$Z <- Z
                              self$n <- length(X)
                              if (self$use_cyclic) {
                                self$X2 <- lapply(X, function(x) self$cyclic_shift$cs(x, forward = TRUE))
                              } else {
                                self$X2 <- X
                              }
                            },
                            get_item = function(idx) {
                              if (!is.null(self$Z)) {
                                return(list(self$X[[idx]], self$X2[[idx]], self$Y[idx], self$Z[[idx]]))
                              } else {
                                return(list(self$X[[idx]], self$X2[[idx]], self$Y[idx]))
                              }
                            },
                            length = function() {
                              return(self$n)
                            }
                          )
)