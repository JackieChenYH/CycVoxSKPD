source("skpd_tensor_utils_full.R")   
library(R6) 
library(Matrix)

StandardScaler <- R6Class(
  "StandardScaler",
  public = list(
    mean=NULL, sd=NULL,
    fit=function(X){ self$mean<-colMeans(X); self$sd<-apply(X,2,sd); self$sd[self$sd==0]<-1 },
    transform=function(X){ scale(X, center=self$mean, scale=self$sd) },
    fit_transform=function(X){ self$fit(X); self$transform(X) }
  ))

SKPD_LogisticRegressor_Cyclic <- R6Class(
  "SKPD_LogisticRegressor_Cyclic",
  public = list(
    # hyper-params
    p1=NULL,d1=NULL,p2=NULL,d2=NULL,p3=NULL,d3=NULL,
    term=NULL,lmbda_a=NULL,lmbda_b=NULL,lmbda_gamma=NULL,
    alpha=NULL,normalization_A=NULL,normalization_B=NULL,
    use_cyclic=NULL,max_iter=NULL,verbose=NULL,
    
    # sizes & parameters 
    s1=NULL,s2=NULL,A1=NULL,A2=NULL,B1=NULL,B2=NULL,gamma=NULL,
    
    # data 
    n=NULL,X1=NULL,X2=NULL,Y=NULL,Z=NULL,
    X1t=NULL,X2t=NULL,
    scaler_X1=NULL,scaler_X2=NULL,scaler_Z=NULL,
    ten2mat=NULL,cyclic_shift=NULL,shift_operator=NULL,
    
    initialize = function(hparams, Ds=NULL, max_iter=100, verbose=TRUE){
      for(nm in names(hparams)) self[[nm]] <- hparams[[nm]]
      self$max_iter <- max_iter; self$verbose <- verbose
      self$s1 <- self$p1*self$p2*self$p3
      self$s2 <- self$d1*self$d2*self$d3
      self$A1 <- matrix(0,self$term,self$s1)
      self$A2 <- matrix(0,self$term,self$s1)
      self$B1 <- matrix(rnorm(self$term*self$s2,0,0.01), self$term,self$s2)
      self$B2 <- matrix(rnorm(self$term*self$s2,0,0.01), self$term,self$s2)
      
      if(!is.null(Ds)){
        self$n <- length(Ds)
        self$X1 <- lapply(Ds, `[[`, 1)
        self$X2 <- lapply(Ds, `[[`, 2)
        self$Y  <- sapply(Ds, `[[`, 3)
        if(length(Ds[[1]])>3){
          self$Z <- t(sapply(Ds, `[[`, 4))
          self$gamma <- matrix(0, ncol(self$Z), 1)
        }
        self$ten2mat <- Ten2MatOperator$new(self$p1,self$d1,self$p2,self$d2,self$p3,self$d3)
        self$cyclic_shift <- CyclicShiftOperator$new()
        self$shift_operator <- TranslationShiftOperator$new(shift_amount=c(1,1,1))
        
        self$X1t <- array(unlist(lapply(self$X1, function(x) self$ten2mat$forward(x))),
                          dim=c(self$n,self$s1,self$s2))
        if(self$use_cyclic){
          shifted <- lapply(self$X2, function(x) self$cyclic_shift$cs(x, forward=TRUE))
          self$X2t <- array(unlist(lapply(shifted, function(x) self$ten2mat$forward(x))),
                            dim=c(self$n,self$s1,self$s2))
        } else self$X2t <- self$X1t
        
        self$scaler_X1 <- StandardScaler$new()
        self$scaler_X2 <- StandardScaler$new()
        self$X1t <- array(self$scaler_X1$fit_transform(matrix(self$X1t,nrow=self$n)),
                          dim=c(self$n,self$s1,self$s2))
        self$X2t <- array(self$scaler_X2$fit_transform(matrix(self$X2t,nrow=self$n)),
                          dim=c(self$n,self$s1,self$s2))
        if(!is.null(self$Z)){
          self$scaler_Z <- StandardScaler$new()
          self$Z <- self$scaler_Z$fit_transform(self$Z)
        }
      }
    },
    
    logistic_loss = function(y,p){
      p <- pmax(pmin(p,1-1e-15),1e-15)
      -mean(y*log(p)+(1-y)*log(1-p))
    },
    reg_penalty = function(){
      self$lmbda_a*(sum(abs(self$A1))+sum(abs(self$A2))) +
        self$lmbda_b*( self$alpha*(sum(abs(self$B1))+sum(abs(self$B2))) +
                         (1-self$alpha)*(sum(self$B1^2)+sum(self$B2^2)) ) +
        ifelse(is.null(self$gamma), 0, self$lmbda_gamma*sum(abs(self$gamma)))
    },
    
    predict_proba_core = function(X1t, X2t,
                                  A1=self$A1,B1=self$B1,
                                  A2=self$A2,B2=self$B2,
                                  Z=self$Z,gamma=self$gamma,
                                  exclude=NULL, logits=FALSE){
      n <- dim(X1t)[1]; lg <- numeric(n)
      for(i in seq_len(n)){
        s <- 0
        for(r in 1:self$term){
          if(is.null(exclude) || !(exclude %in% c("A1","B1")))
            s <- s + A1[r,] %*% (X1t[i,,] %*% B1[r,])
          if(self$use_cyclic && (is.null(exclude) || !(exclude %in% c("A2","B2"))))
            s <- s + A2[r,] %*% (X2t[i,,] %*% B2[r,])
        }
        if(!is.null(Z) && (is.null(exclude) || exclude!="gamma"))
          s <- s + Z[i,] %*% gamma
        lg[i] <- s
      }
      if(logits) return(lg)
      1/(1+exp(-pmax(pmin(lg,50),-50)))
    },
    
    update_B1 = function(){
      F1 <- matrix(0, self$n, self$s2*self$term)
      for(i in 1:self$n)
        for(r in 1:self$term){
          idx <- ((r-1)*self$s2+1):(r*self$s2)
          F1[i,idx] <- t(self$X1t[i,,]) %*% self$A1[r,]
        }
      scaler <- StandardScaler$new(); F1s <- scaler$fit_transform(F1)
      off <- if(self$use_cyclic) self$predict_proba_core(self$X1t,self$X2t,exclude="B1",logits=TRUE) else rep(0,self$n)
      if(!is.null(self$Z)) off <- off + self$Z %*% self$gamma
      if(max(abs(off))>1e-10) off <- off/max(abs(off))
      obj <- function(par){
        lg <- F1s %*% par + off
        pr <- 1/(1+exp(-pmax(pmin(lg,50),-50)))
        loss <- -mean(self$Y*log(pr+1e-15)+(1-self$Y)*log(1-pr+1e-15))
        pen  <- self$lmbda_b*sum(self$alpha*abs(par)+(1-self$alpha)*par^2)
        loss+pen
      }
      res <- optim(as.vector(self$B1), obj, method="L-BFGS-B")
      self$B1 <- matrix(res$par, self$term, self$s2)
      if(self$normalization_B)
        self$B1 <- t(apply(self$B1,1,function(v) if(sum(v^2)>0) v/sqrt(sum(v^2)) else v))
    },
    
    update_A1 = function(){
      G1 <- matrix(0, self$n, self$s1*self$term)
      for(i in 1:self$n)
        for(r in 1:self$term){
          idx <- ((r-1)*self$s1+1):(r*self$s1)
          G1[i,idx] <- self$X1t[i,,] %*% self$B1[r,]
        }
      scaler <- StandardScaler$new(); G1s <- scaler$fit_transform(G1)
      off <- if(self$use_cyclic) self$predict_proba_core(self$X1t,self$X2t,exclude="A1",logits=TRUE) else rep(0,self$n)
      if(!is.null(self$Z)) off <- off + self$Z %*% self$gamma
      if(max(abs(off))>1e-10) off <- off/max(abs(off))
      obj <- function(par){
        lg <- G1s %*% par + off
        pr <- 1/(1+exp(-pmax(pmin(lg,50),-50)))
        loss <- -mean(self$Y*log(pr+1e-15)+(1-self$Y)*log(1-pr+1e-15))
        pen  <- self$lmbda_a*sum(abs(par))
        loss+pen
      }
      res <- optim(as.vector(self$A1), obj, method="L-BFGS-B")
      self$A1 <- matrix(res$par, self$term, self$s1)
      if(self$normalization_A)
        self$A1 <- t(apply(self$A1,1,function(v) if(sum(v^2)>0) v/sqrt(sum(v^2)) else v))
    },
    
    update_B2 = function(){
      if(!self$use_cyclic){ self$B2[,] <- 0; return() }
      F2 <- matrix(0, self$n, self$s2*self$term)
      for(i in 1:self$n)
        for(r in 1:self$term){
          idx <- ((r-1)*self$s2+1):(r*self$s2)
          F2[i,idx] <- t(self$X2t[i,,]) %*% self$A2[r,]
        }
      scaler <- StandardScaler$new(); F2s <- scaler$fit_transform(F2)
      off <- self$predict_proba_core(self$X1t,self$X2t,exclude="B2",logits=TRUE)
      if(!is.null(self$Z)) off <- off + self$Z %*% self$gamma
      if(max(abs(off))>1e-10) off <- off/max(abs(off))
      obj <- function(par){
        lg <- F2s %*% par + off
        pr <- 1/(1+exp(-pmax(pmin(lg,50),-50)))
        loss <- -mean(self$Y*log(pr+1e-15)+(1-self$Y)*log(1-pr+1e-15))
        pen  <- self$lmbda_b*sum(self$alpha*abs(par)+(1-self$alpha)*par^2)
        loss+pen
      }
      res <- optim(as.vector(self$B2), obj, method="L-BFGS-B")
      self$B2 <- matrix(res$par, self$term, self$s2)
      if(self$normalization_B)
        self$B2 <- t(apply(self$B2,1,function(v) if(sum(v^2)>0) v/sqrt(sum(v^2)) else v))
    },
    
    update_A2 = function(){
      if(!self$use_cyclic){ self$A2[,] <- 0; return() }
      G2 <- matrix(0, self$n, self$s1*self$term)
      for(i in 1:self$n)
        for(r in 1:self$term){
          idx <- ((r-1)*self$s1+1):(r*self$s1)
          G2[i,idx] <- self$X2t[i,,] %*% self$B2[r,]
        }
      scaler <- StandardScaler$new(); G2s <- scaler$fit_transform(G2)
      off <- self$predict_proba_core(self$X1t,self$X2t,exclude="A2",logits=TRUE)
      if(!is.null(self$Z)) off <- off + self$Z %*% self$gamma
      if(max(abs(off))>1e-10) off <- off/max(abs(off))
      obj <- function(par){
        lg <- G2s %*% par + off
        pr <- 1/(1+exp(-pmax(pmin(lg,50),-50)))
        loss <- -mean(self$Y*log(pr+1e-15)+(1-self$Y)*log(1-pr+1e-15))
        pen  <- self$lmbda_a*sum(abs(par))
        loss+pen
      }
      res <- optim(as.vector(self$A2), obj, method="L-BFGS-B")
      self$A2 <- matrix(res$par, self$term, self$s1)
      if(self$normalization_A)
        self$A2 <- t(apply(self$A2,1,function(v) if(sum(v^2)>0) v/sqrt(sum(v^2)) else v))
    },
    
    update_gamma = function(){
      if(is.null(self$Z)) return()
      off <- self$predict_proba_core(self$X1t,self$X2t,logits=TRUE)
      if(max(abs(off))>1e-10) off <- off/max(abs(off))
      obj <- function(gpar){
        lg <- self$Z %*% gpar + off
        pr <- 1/(1+exp(-pmax(pmin(lg,50),-50)))
        loss <- -mean(self$Y*log(pr+1e-15)+(1-self$Y)*log(1-pr+1e-15))
        loss + self$lmbda_gamma*sum(abs(gpar))
      }
      init <- if(is.null(self$gamma)) rnorm(ncol(self$Z),0,0.1) else as.vector(self$gamma)
      res <- optim(init, obj, method="L-BFGS-B")
      self$gamma <- matrix(res$par,ncol=1)
    },
    

    fit = function(){
      w1 <- Reduce("+",lapply(seq_len(self$n), function(i) self$Y[i]*self$X1t[i,,]))/self$n
      self$A1 <- t(svd(w1)$u[,1:self$term,drop=FALSE])
      w2 <- Reduce("+",lapply(seq_len(self$n), function(i) self$Y[i]*self$X2t[i,,]))/self$n
      self$A2 <- t(svd(w2)$u[,1:self$term,drop=FALSE])
      if(!self$use_cyclic){ self$A2[,] <- 0; self$B2[,] <- 0 }
      prev <- Inf
      for(it in 1:self$max_iter){
        if(self$verbose) cat(sprintf("Iter %d\\n", it))
        self$update_B1(); self$update_A1()
        if(self$use_cyclic){ self$update_B2(); self$update_A2() }
        if(!is.null(self$Z)) self$update_gamma()
        pr <- self$predict_proba_core(self$X1t,self$X2t)
        loss <- self$logistic_loss(self$Y,pr)+self$reg_penalty()
        if(self$verbose) cat(sprintf("Loss %.6f\\n", loss))
        if(it>5 && abs(prev-loss)/max(prev,1e-8)<1e-5) break
        prev <- loss
      }
      invisible(loss)
    },
    
    predict_proba = function(X1,X2=NULL){
      n <- length(X1)
      X1t <- array(unlist(lapply(X1, function(x) self$ten2mat$forward(x))),
                   dim=c(n,self$s1,self$s2))
      X1t <- array(self$scaler_X1$transform(matrix(X1t,nrow=n)),
                   dim=c(n,self$s1,self$s2))
      if(self$use_cyclic){
        X2t <- array(unlist(lapply(X2, function(x) self$ten2mat$forward(
          self$cyclic_shift$cs(x, forward=TRUE)))),
          dim=c(n,self$s1,self$s2))
      } else X2t <- X1t
      self$predict_proba_core(X1t,X2t)
    }
  ))
