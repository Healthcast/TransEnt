\name{TransferEntropy-package}
\alias{TransferEntropy-package}
\docType{package}
\title{
\packageTitle{TransferEntropy}
}
\description{
\packageDescription{TransferEntropy}
The main function \code{computeTE} calculates the transfer entropy from one continuous-valued random process to another. Two calculaton methods are offered. 

Trans Entrop is an information theoric measure that quantifies the statistical coherence between systems
evolving in time. Unlike mutual information, transfer entropy is more adequate for indicating the direction of the information flow. 
\deqn{ T_{X->Y} = \Sigma p(Y_{n+1},Y_{n}^{(k)},X_{n}^{(l)}) \log { \frac{p(Y_{n+1}|Y_{n}^{(k)},X_{n}^{(l)})}{p(Y_{n+1}|Y_{n}^{(k)})} } }

TE can be computed in a straightforward way for discrete valued random processes by counting. This does not directly generalize to continuous valued variables as continuous random processes seldom revisit prior values. Naive binning may distort the estimate of the underlying probability distribution. Hence the development of more statistically robust estimation methods such as the generalized correlation sum or the Kraskov estimator, which are used here. 

Please see the references for additional background on the derivation and intepretation of the transfer entropy and the issues of estimating ratios of probability distributions from observations of continuously distributed random variables.
}

\author{
\packageAuthor{TransferEntropy}
Maintainer: \packageMaintainer{TransferEntropy}
}

\references{
Measuring Information Transfer. Thomas Schreiber. 2000.

Information Transfer in Continuous Processes. A. Kaiser, T. Schreiber. 2002.

Estimating Mutual Information. Alexander Kraskov, Harald Stoegbauer, Peter Grassberger. 2004.

A Recipe for the Estimation of Information Flow in a Dynamical System. Deniz Gencaga, Kevin H. Knuth, William B. Rossow. 2015.
}