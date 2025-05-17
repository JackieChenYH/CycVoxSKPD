
# skpd_tensor_utils.R
# --------------------------------------------------------------
# R translations of:
#   * Ten2MatOperator
#   * Mat2TenOperator
#   * TranslationShiftOperator
#   * CyclicShiftOperator
#   * CS_SKPDDataSet
# --------------------------------------------------------------
library(R6)

# ---------- Ten2MatOperator ----------
Ten2MatOperator <- R6Class("Ten2MatOperator",
  public = list(
    p1 = NULL, d1 = NULL, p2 = NULL, d2 = NULL, p3 = 1, d3 = 1,

    initialize = function(p1, d1, p2, d2, p3 = 1, d3 = 1) {
      self$p1 <- p1; self$d1 <- d1
      self$p2 <- p2; self$d2 <- d2
      self$p3 <- p3; self$d3 <- d3
    },

    forward = function(C) {
      if (length(dim(C)) == 2) {
        m <- nrow(C); n <- ncol(C)
        stopifnot(m == self$p1 * self$d1, n == self$p2 * self$d2)
        RC <- list()
        for (i in 0:(self$p1-1)) {
          for (j in 0:(self$p2-1)) {
            block <- C[(i*self$d1+1):((i+1)*self$d1),
                       (j*self$d2+1):((j+1)*self$d2)]
            RC[[length(RC)+1]] <- as.vector(block)
          }
        }
        return(do.call(rbind, RC))
      } else if (length(dim(C)) == 3) {
        dims <- dim(C)
        stopifnot(dims[1] == self$p1*self$d1,
                  dims[2] == self$p2*self$d2,
                  dims[3] == self$p3*self$d3)
        RC <- list()
        for (i in 0:(self$p1-1)) {
          for (j in 0:(self$p2-1)) {
            for (k in 0:(self$p3-1)) {
              block <- C[(i*self$d1+1):((i+1)*self$d1),
                         (j*self$d2+1):((j+1)*self$d2),
                         (k*self$d3+1):((k+1)*self$d3)]
              RC[[length(RC)+1]] <- as.vector(block)
            }
          }
        }
        return(do.call(rbind, RC))
      } else stop("Input must be 2D or 3D.")
    }
  )
)

# ---------- Mat2TenOperator ----------
Mat2TenOperator <- R6Class("Mat2TenOperator",
  public = list(
    p1 = NULL, d1 = NULL, p2 = NULL, d2 = NULL, p3 = 1, d3 = 1,

    initialize = function(p1, d1, p2, d2, p3 = 1, d3 = 1) {
      self$p1 <- p1; self$d1 <- d1
      self$p2 <- p2; self$d2 <- d2
      self$p3 <- p3; self$d3 <- d3
    },

    forward = function(RC) {
      if (self$p3 == 1 && self$d3 == 1) {
        # matrix case
        stopifnot(ncol(RC) == self$d1*self$d2)
        C <- matrix(0, self$p1*self$d1, self$p2*self$d2)
        for (idx in 0:(nrow(RC)-1)) {
          ith <- idx %/% self$p2
          jth <- idx %% self$p2
          block <- matrix(RC[idx+1, ], self$d1, self$d2)
          C[(ith*self$d1+1):((ith+1)*self$d1),
            (jth*self$d2+1):((jth+1)*self$d2)] <- block
        }
        return(C)
      } else {
        # tensor case
        C <- array(0, dim=c(self$p1*self$d1, self$p2*self$d2, self$p3*self$d3))
        p2p3 <- self$p2*self$p3
        for (idx in 0:(nrow(RC)-1)) {
          ith <- idx %/% p2p3
          rem <- idx %% p2p3
          jth <- rem %/% self$p3
          kth <- rem %% self$p3
          block <- array(RC[idx+1, ], dim=c(self$d1,self$d2,self$d3))
          C[(ith*self$d1+1):((ith+1)*self$d1),
            (jth*self$d2+1):((jth+1)*self$d2),
            (kth*self$d3+1):((kth+1)*self$d3)] <- block
        }
        return(C)
      }
    }
  )
)

# ---------- TranslationShiftOperator ----------
TranslationShiftOperator <- R6Class("TranslationShiftOperator",
  public = list(
    shift_amount = NULL,

    initialize = function(shift_amount = c(1,1,1)) {
      self$shift_amount <- shift_amount
    },

    cs = function(tensor, forward = TRUE) {
      s <- if (forward) self$shift_amount else -self$shift_amount
      dims <- dim(tensor)
      out <- array(0, dims)
      if (length(dims) == 2) {
        r <- dims[1]; c <- dims[2]
        if (all(s >= 0)) {
          out[(s[1]+1):r, (s[2]+1):c] <- tensor[1:(r-s[1]), 1:(c-s[2])]
        } else {
          out[1:(r+s[1]), 1:(c+s[2])] <- tensor[(1-s[1]):r, (1-s[2]):c]
        }
      } else if (length(dims) == 3) {
        r <- dims[1]; c <- dims[2]; d <- dims[3]
        if (all(s >= 0)) {
          out[(s[1]+1):r,(s[2]+1):c,(s[3]+1):d] <- tensor[1:(r-s[1]),1:(c-s[2]),1:(d-s[3])]
        } else {
          out[1:(r+s[1]),1:(c+s[2]),1:(d+s[3])] <- tensor[(1-s[1]):r,(1-s[2]):c,(1-s[3]):d]
        }
      } else stop("Tensor must be 2D or 3D.")
      out
    }
  )
)

# ---------- CyclicShiftOperator ----------
CyclicShiftOperator <- R6Class("CyclicShiftOperator",
  public = list(
    cs = function(tensor, forward = TRUE) {
      dims <- dim(tensor)
      if (length(dims) == 2) {
        mid <- floor(dims/2)
        idx1 <- if (forward) c((dims[1]-mid[1]+1):dims[1], 1:(dims[1]-mid[1])) else c((mid[1]+1):dims[1], 1:mid[1])
        idx2 <- if (forward) c((dims[2]-mid[2]+1):dims[2], 1:(dims[2]-mid[2])) else c((mid[2]+1):dims[2], 1:mid[2])
        tensor[idx1, idx2]
      } else if (length(dims) == 3) {
        mid <- floor(dims/2)
        idx1 <- if (forward) c((dims[1]-mid[1]+1):dims[1], 1:(dims[1]-mid[1])) else c((mid[1]+1):dims[1], 1:mid[1])
        idx2 <- if (forward) c((dims[2]-mid[2]+1):dims[2], 1:(dims[2]-mid[2])) else c((mid[2]+1):dims[2], 1:mid[2])
        idx3 <- if (forward) c((dims[3]-mid[3]+1):dims[3], 1:(dims[3]-mid[3])) else c((mid[3]+1):dims[3], 1:mid[3])
        tensor[idx1, idx2, idx3]
      } else stop("Only 2D or 3D tensors supported.")
    }
  )
)

# ---------- CS_SKPDDataSet ----------
CS_SKPDDataSet <- R6Class("CS_SKPDDataSet",
  public = list(
    p1=NULL,d1=NULL,p2=NULL,d2=NULL,p3=NULL,d3=NULL,
    use_cyclic=NULL, cyclic_shift=NULL,
    X=NULL, X2=NULL, Y=NULL, Z=NULL, n=NULL,

    initialize = function(hparams, X, Y, Z=NULL) {
      # params
      self$p1 <- hparams$p1; self$d1 <- hparams$d1
      self$p2 <- hparams$p2; self$d2 <- hparams$d2
      self$p3 <- hparams$p3; self$d3 <- hparams$d3
      self$use_cyclic <- ifelse(is.null(hparams$use_cyclic), TRUE, hparams$use_cyclic)
      self$cyclic_shift <- CyclicShiftOperator$new()

      stopifnot(length(X)==length(Y))
      self$X <- X; self$Y <- Y; self$Z <- Z
      self$n <- length(X)

      if (self$use_cyclic) {
        self$X2 <- lapply(self$X, function(x) self$cyclic_shift$cs(x, TRUE))
      } else self$X2 <- self$X
    },

    length = function() self$n,

    get_item = function(idx) {
      if (!is.null(self$Z))
        list(self$X[[idx]], self$X2[[idx]], self$Y[[idx]], self$Z[[idx]])
      else
        list(self$X[[idx]], self$X2[[idx]], self$Y[[idx]])
    }
  )
)
