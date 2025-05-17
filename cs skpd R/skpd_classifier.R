
# skpd_classifier.R
# R translation of classifier_optimize_spatialShift.py
library(R6)
library(Matrix)

SKPD_LogisticRegressor_Cyclic <- R6Class("SKPD_LogisticRegressor_Cyclic",
  public = list(
    # Hyperparameters
    p1 = NULL, d1 = NULL, p2 = NULL, d2 = NULL, p3 = NULL, d3 = NULL,
    term = NULL,
    lmbda_a = NULL, lmbda_b = NULL, lmbda_gamma = NULL,
    alpha = NULL,
    normalization_A = NULL, normalization_B = NULL,
    use_cyclic = NULL,
    max_iter = NULL, verbose = NULL,

    # Model parameters
    s1 = NULL, s2 = NULL,
    A1 = NULL, A2 = NULL, B1 = NULL, B2 = NULL, gamma = NULL,

    # Data and helpers
    n = NULL,   
    X1 = NULL, X2 = NULL, Y = NULL, Z = NULL,
    X1_transformed = NULL, X2_transformed = NULL,
    scaler_X1 = NULL, scaler_X2 = NULL, scaler_Z = NULL,
    ten2mat = NULL, cyclic_shift = NULL,

    initialize = function(hparams, Ds = NULL, max_iter = 100, verbose = TRUE) {
      # Assign hyperparams
      for (nm in names(hparams)) self[[nm]] <- hparams[[nm]]
      self$max_iter <- max_iter; self$verbose <- verbose

      # Derived sizes
      self$s1 <- self$p1 * self$p2 * self$p3
      self$s2 <- self$d1 * self$d2 * self$d3

      # Init parameters
      self$A1 <- matrix(0, self$term, self$s1)
      self$A2 <- matrix(0, self$term, self$s1)
      self$B1 <- matrix(rnorm(self$term * self$s2, 0, 0.01), self$term)
      self$B2 <- matrix(rnorm(self$term * self$s2, 0, 0.01), self$term)

      # Helpers
      self$ten2mat <- Ten2MatOperator$new(self$p1, self$d1, self$p2, self$d2, self$p3, self$d3)
      self$cyclic_shift <- CyclicShiftOperator$new()

      # Load dataset if provided
      if (!is.null(Ds)) {
        self$n <- length(Ds)
        self$X1 <- lapply(Ds, function(d) d[[1]])
        self$X2 <- lapply(Ds, function(d) d[[2]])
        self$Y <- sapply(Ds, function(d) d[[3]])
        if (length(Ds[[1]]) > 3) self$Z <- lapply(Ds, function(d) d[[4]])

        # Transform tensors to matrices
        self$X1_transformed <- lapply(self$X1, function(x) self$ten2mat$forward(x))
        if (self$use_cyclic) {
          shifted <- lapply(self$X2, function(x) self$cyclic_shift$cs(x, TRUE))
          self$X2_transformed <- lapply(shifted, function(x) self$ten2mat$forward(x))
        } else self$X2_transformed <- self$X1_transformed

        # Scale
        scale_mat <- function(mat_list) {
          flat <- do.call(rbind, lapply(mat_list, as.vector))
          mu <- colMeans(flat); sdv <- apply(flat, 2, sd); sdv[sdv == 0] <- 1
          scaled <- scale(flat, mu, sdv)
          lapply(seq_len(nrow(scaled)), function(i) matrix(scaled[i, ], self$s1, self$s2))
        }
        self$X1_transformed <- scale_mat(self$X1_transformed)
        self$X2_transformed <- scale_mat(self$X2_transformed)
        if (!is.null(self$Z)) {
          Zm <- do.call(rbind, self$Z)
          mu <- colMeans(Zm); sdv <- apply(Zm, 2, sd); sdv[sdv == 0] <- 1
          self$Z <- lapply(seq_len(nrow(Zm)), function(i) as.numeric((Zm[i, ] - mu) / sdv))
          self$gamma <- rep(0, ncol(Zm))
        }
      }
    },

    logistic_loss = function(y, p) {
      p <- pmin(pmax(p, 1e-15), 1 - 1e-15)
      -mean(y * log(p) + (1 - y) * log(1 - p))
    },

    regularization_penalty = function() {
      pa <- self$lmbda_a * (sum(abs(self$A1)) + sum(abs(self$A2)))
      pb <- self$lmbda_b * (self$alpha * (sum(abs(self$B1)) + sum(abs(self$B2))) +
            (1 - self$alpha) * (sum(self$B1^2) + sum(self$B2^2)))
      pg <- if (!is.null(self$gamma)) self$lmbda_gamma * sum(abs(self$gamma)) else 0
      pa + pb + pg
    },

    predict_proba_internal = function(A1, B1, A2, B2, gamma, exclude = NULL, logit_only = FALSE) {
      logits <- numeric(self$n)
      for (i in seq_len(self$n)) {
        lo <- 0
        for (r in seq_len(self$term)) {
          if (is.null(exclude) || !(exclude %in% c("A1", "B1"))) 
            lo <- lo + sum(A1[r, ] * (self$X1_transformed[[i]] %*% B1[r, ]))
          if (self$use_cyclic && (is.null(exclude) || !(exclude %in% c("A2","B2"))))
            lo <- lo + sum(A2[r, ] * (self$X2_transformed[[i]] %*% B2[r, ]))
        }
        if (!is.null(gamma) && (is.null(exclude) || exclude != "gamma") && !is.null(self$Z))
          lo <- lo + sum(self$Z[[i]] * gamma)
        logits[i] <- lo
      }
      if (logit_only) return(logits)
      1 / (1 + exp(-pmin(pmax(logits, -50), 50)))
    },

    update_matrix = function(target) {
      mat <- self[[target]]
      other <- self[[paste0(ifelse(substr(target,1,1)=="A","B","A"), substr(target,2,2))]]
      Xlist <- if (substr(target,2,2)=="1") self$X1_transformed else self$X2_transformed
      cols <- ncol(other)
      G <- matrix(0, self$n, cols * self$term)
      for (i in seq_len(self$n)) {
        for (r in seq_len(self$term)) {
          idx <- ((r-1)*cols+1):(r*cols)
          G[i, idx] <- if (substr(target,1,1)=="A")
              Xlist[[i]] %*% other[r, ] else t(Xlist[[i]]) %*% other[r, ]
        }
      }
      G <- scale(G)
      offset <- self$predict_proba_internal(self$A1, self$B1, self$A2, self$B2, self$gamma, exclude = target, logit_only = TRUE)
      if (!is.null(self$Z)) offset <- offset + sapply(self$Z, function(z) sum(z * self$gamma))
      if (max(abs(offset)) > 1e-10) offset <- offset / max(abs(offset))
      obj <- function(par) {
        M <- matrix(par, self$term)
        logits <- G %*% par + offset
        p <- 1/(1+exp(-pmin(pmax(logits,-50),50)))
        loss <- self$logistic_loss(self$Y, p)
        pen <- if (substr(target,1,1)=="A") self$lmbda_a * sum(abs(M))
               else self$lmbda_b * sum(self$alpha * abs(M) + (1-self$alpha)*M^2)
        loss + pen
      }
      res <- optim(par = as.vector(mat), fn = obj, method = "L-BFGS-B")
      self[[target]] <- matrix(res$par, self$term)
      if (substr(target,1,1)=="A" && self$normalization_A)
        self[[target]] <- t(apply(self[[target]],1, function(v) v/sqrt(sum(v^2)+1e-10)))
      if (substr(target,1,1)=="B" && self$normalization_B)
        self[[target]] <- t(apply(self[[target]],1, function(v) v/sqrt(sum(v^2)+1e-10)))
    },

    update_gamma = function() {
      if (is.null(self$Z)) return()
      Zm <- do.call(rbind, self$Z)
      offset <- self$predict_proba_internal(self$A1,self$B1,self$A2,self$B2,self$gamma,logit_only=TRUE)
      if (max(abs(offset))>1e-10) offset <- offset/max(abs(offset))
      obj <- function(g) {
        logits <- Zm %*% g + offset
        p <- 1/(1+exp(-pmin(pmax(logits,-50),50)))
        self$logistic_loss(self$Y,p) + self$lmbda_gamma*sum(abs(g))
      }
      self$gamma <- optim(par=self$gamma, fn=obj, method="L-BFGS-B")$par
    },

    fit = function() {
      # SVD init
      W1 <- Reduce("+", lapply(seq_len(self$n), function(i) self$Y[i]*self$X1_transformed[[i]]))/self$n
      U1 <- svd(W1)$u; k1 <- min(ncol(U1), self$term)
      self$A1 <- t(U1[,1:k1]); if (k1<self$term) self$A1 <- rbind(self$A1, matrix(0,self$term-k1,self$s1))
      W2 <- Reduce("+", lapply(seq_len(self$n), function(i) self$Y[i]*self$X2_transformed[[i]]))/self$n
      U2 <- svd(W2)$u; k2 <- min(ncol(U2), self$term)
      self$A2 <- t(U2[,1:k2]); if (k2<self$term) self$A2 <- rbind(self$A2, matrix(0,self$term-k2,self$s1))
      if (!self$use_cyclic) { self$A2[,] <- 0; self$B2[,] <- 0 }

      prev <- Inf
      for (it in 1:self$max_iter) {
        if (self$verbose) cat(sprintf("Iter %d
", it))
        self$update_matrix("B1"); self$update_matrix("A1")
        if (self$use_cyclic) { self$update_matrix("B2"); self$update_matrix("A2") }
        if (!is.null(self$Z)) self$update_gamma()
        p <- self$predict_proba_internal(self$A1,self$B1,self$A2,self$B2,self$gamma)
        loss <- self$logistic_loss(self$Y,p) + self$regularization_penalty()
        if (self$verbose) cat(sprintf("Loss %.6f
", loss))
        if (is.finite(prev) && abs(prev - loss)/max(prev, 1e-8) < 1e-5)
          break
        prev <- loss
      }
    },

    predict_proba = function(X1, X2=NULL) {
      Xt1 <- lapply(X1, function(x) self$ten2mat$forward(x))
      Xt1 <- lapply(Xt1, function(m) matrix((as.vector(m)-self$scaler_X1$mean)/self$scaler_X1$sd, self$s1,self$s2))
      if (self$use_cyclic && !is.null(X2)) {
        Xt2 <- lapply(X2, function(x) self$ten2mat$forward(self$cyclic_shift$cs(x,TRUE)))
        Xt2 <- lapply(Xt2, function(m) matrix((as.vector(m)-self$scaler_X2$mean)/self$scaler_X2$sd, self$s1,self$s2))
      } else Xt2 <- Xt1
      old_X1 <- self$X1_transformed; old_X2 <- self$X2_transformed
      self$X1_transformed <- Xt1; self$X2_transformed <- Xt2
      self$n <- length(Xt1)
      probs <- self$predict_proba_internal(self$A1,self$B1,self$A2,self$B2,self$gamma)
      self$X1_transformed <- old_X1; self$X2_transformed <- old_X2; self$n <- length(old_X1)
      probs
    }
))
