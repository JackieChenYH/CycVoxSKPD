library(R6)

Ten2MatOperator <- R6Class(
  "Ten2MatOperator",
  public = list(
    p1 = NULL, d1 = NULL, p2 = NULL, d2 = NULL, p3 = NULL, d3 = NULL,
    initialize = function(p1, d1, p2, d2, p3 = 1, d3 = 1) {
      self$p1 <- p1; self$d1 <- d1
      self$p2 <- p2; self$d2 <- d2
      self$p3 <- p3; self$d3 <- d3
    },
    forward = function(C) {
      if (length(dim(C)) == 2) {
        m <- nrow(C); n <- ncol(C)
        stopifnot(m == self$p1 * self$d1, n == self$p2 * self$d2)
        RC <- vector("list", self$p1 * self$p2)
        idx <- 1
        for (i in 0:(self$p1 - 1))
          for (j in 0:(self$p2 - 1)) {
            blk <- C[(self$d1*i+1):(self$d1*(i+1)),
                     (self$d2*j+1):(self$d2*(j+1))]
            RC[[idx]] <- as.vector(t(blk))
            idx <- idx + 1
          }
        return(t(do.call(rbind, RC)))
      } else if (length(dim(C)) == 3) {
        dd <- dim(C)
        stopifnot(dd[1] == self$p1*self$d1,
                  dd[2] == self$p2*self$d2,
                  dd[3] == self$p3*self$d3)
        RC <- vector("list", self$p1*self$p2*self$p3)
        idx <- 1
        for (i in 0:(self$p1-1))
          for (j in 0:(self$p2-1))
            for (k in 0:(self$p3-1)) {
              blk <- C[(self$d1*i+1):(self$d1*(i+1)),
                       (self$d2*j+1):(self$d2*(j+1)),
                       (self$d3*k+1):(self$d3*(k+1))]
              RC[[idx]] <- as.vector(t(blk)); idx <- idx+1
            }
        return(t(do.call(rbind, RC)))
      } else stop("Input must be 2D or 3D.")
    }
  ))

Mat2TenOperator <- R6Class(
  "Mat2TenOperator",
  public = list(
    p1=NULL,d1=NULL,p2=NULL,d2=NULL,p3=NULL,d3=NULL,
    initialize=function(p1,d1,p2,d2,p3=1,d3=1){
      self$p1<-p1;self$d1<-d1;self$p2<-p2;self$d2<-d2;self$p3<-p3;self$d3<-d3
    },
    forward=function(RC){
      if(self$p3==1 && self$d3==1){
        C <- matrix(0, self$p1*self$d1, self$p2*self$d2)
        for(i in 1:nrow(RC)){
          blk <- matrix(RC[i,], self$d1, self$d2)
          ith <- (i-1)%/%self$p2; jth <- (i-1)%%self$p2
          C[(self$d1*ith+1):(self$d1*(ith+1)),
            (self$d2*jth+1):(self$d2*(jth+1))] <- blk
        }
        C
      } else {
        C <- array(0, dim=c(self$p1*self$d1, self$p2*self$d2, self$p3*self$d3))
        p2p3 <- self$p2*self$p3
        for(i in 1:nrow(RC)){
          blk <- array(RC[i,], dim=c(self$d1,self$d2,self$d3))
          ith <- (i-1)%/%p2p3
          rem <- (i-1)%%p2p3
          jth <- rem%/%self$p3
          kth <- rem%%self$p3
          C[(self$d1*ith+1):(self$d1*(ith+1)),
            (self$d2*jth+1):(self$d2*(jth+1)),
            (self$d3*kth+1):(self$d3*(kth+1))] <- blk
        }
        C
      }
    }
  ))


TranslationShiftOperator <- R6Class(
  "TranslationShiftOperator",
  public = list(
    shift_amount=NULL,
    initialize=function(shift_amount=c(1,1,1)){ self$shift_amount <- shift_amount },
    cs=function(tensor, forward=TRUE, ...){
      if(forward) private$shift(tensor, self$shift_amount)
      else        private$shift(tensor, -self$shift_amount)
    }
  ),
  private = list(
    shift=function(tensor, sh){
      if(length(dim(tensor))==2){
        s1<-sh[1];s2<-sh[2];p1<-nrow(tensor);p2<-ncol(tensor)
        out<-matrix(0,p1,p2)
        out[1:(p1-s1),1:(p2-s2)] <- tensor[(s1+1):p1,(s2+1):p2]
        out
      } else {
        s1<-sh[1];s2<-sh[2];s3<-sh[3]
        p1<-dim(tensor)[1];p2<-dim(tensor)[2];p3<-dim(tensor)[3]
        out<-array(0,dim=dim(tensor))
        out[1:(p1-s1),1:(p2-s2),1:(p3-s3)] <- tensor[(s1+1):p1,(s2+1):p2,(s3+1):p3]
        out
      }
    }
  ))


CyclicShiftOperator <- R6Class(
  "CyclicShiftOperator",
  public = list(
    cs = function(tensor, forward=TRUE,
                  original_dims=NULL, scaling_factors=NULL, custom_shift_amounts=NULL){
      if(length(dim(tensor))==2){
        p1<-nrow(tensor); p2<-ncol(tensor)
        mid1<-if(!is.null(original_dims)) original_dims[1]%/%2 else p1%/%2
        mid2<-if(!is.null(original_dims)) original_dims[2]%/%2 else p2%/%2
        idx1 <- 1:p1; idx2 <- 1:p2
        if(forward){
          idx1 <- c(tail(idx1,mid1), head(idx1,p1-mid1))
          idx2 <- c(tail(idx2,mid2), head(idx2,p2-mid2))
        } else {
          idx1 <- c(tail(idx1,p1-mid1), head(idx1,mid1))
          idx2 <- c(tail(idx2,p2-mid2), head(idx2,mid2))
        }
        tensor[idx1, idx2]
      } else {
        p1<-dim(tensor)[1]; p2<-dim(tensor)[2]; p3<-dim(tensor)[3]
        mid1<-if(!is.null(original_dims)) original_dims[1]%/%2 else p1%/%2
        mid2<-if(!is.null(original_dims)) original_dims[2]%/%2 else p2%/%2
        mid3<-if(!is.null(original_dims)) original_dims[3]%/%2 else p3%/%2
        idx1 <- 1:p1; idx2 <- 1:p2; idx3 <- 1:p3
        if(forward){
          idx1 <- c(tail(idx1,mid1), head(idx1,p1-mid1))
          idx2 <- c(tail(idx2,mid2), head(idx2,p2-mid2))
          idx3 <- c(tail(idx3,mid3), head(idx3,p3-mid3))
        } else {
          idx1 <- c(tail(idx1,p1-mid1), head(idx1,mid1))
          idx2 <- c(tail(idx2,p2-mid2), head(idx2,mid2))
          idx3 <- c(tail(idx3,p3-mid3), head(idx3,mid3))
        }
        tensor[idx1, idx2, idx3]
      }
    }
  ))


CS_SKPDDataSet <- R6Class(
  "CS_SKPDDataSet",
  public = list(
    use_cyclic=NULL, cyclic_shift=NULL,
    X=NULL, X2=NULL, Y=NULL, Z=NULL, n=NULL,
    initialize = function(hparams, X, Y, Z=NULL){
      stopifnot(length(X)==length(Y))
      self$use_cyclic <- ifelse(is.null(hparams$use_cyclic), TRUE, hparams$use_cyclic)
      self$cyclic_shift <- CyclicShiftOperator$new()
      self$X <- X; self$Y <- Y; self$Z <- Z; self$n <- length(X)
      self$X2 <- if(self$use_cyclic)
        lapply(X, function(x) self$cyclic_shift$cs(x, forward=TRUE))
      else X
    },
    length = function() self$n,
    get = function(idx){
      if(!is.null(self$Z))
        list(self$X[[idx]], self$X2[[idx]], self$Y[[idx]], self$Z[idx,])
      else
        list(self$X[[idx]], self$X2[[idx]], self$Y[[idx]])
    }
  ))
