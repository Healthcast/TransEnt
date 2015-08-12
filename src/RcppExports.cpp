// This file was generated by Rcpp::compileAttributes
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393
#include <vector>
#include <string>
#include <Rcpp.h>
#include <ANN/ANN.h>

using namespace Rcpp;

int compute_TE(double& TE, std::vector<double>&X, std::vector<double>&Y,
		       int embedding,int k, std::string method, double epsDistance=-1, bool addNoise = false);

RcppExport SEXP Rcpp_ComputeTE(SEXP Rx, SEXP Ry, SEXP Re, SEXP Rk, SEXP Rm, SEXP RepsDist)
{
	List ret;
	double TE;
	NumericVector xx(clone(Rx));
	NumericVector yy(clone(Ry));
	std::vector<double> X(xx.begin(),xx.end());
	std::vector<double> Y(yy.begin(),yy.end());
	std::string method = as<std::string>(Rm);
	int embedding = as<int>(Re);
	int k = as<int>(Rk);
	double epsDist = as<double>(RepsDist);
	try{
		Rcpp::wrap(compute_TE(TE,X,Y,embedding,k,method,epsDist,false));
	}
	catch(std::invalid_argument& e)
	{
		forward_exception_to_r(e);
	}
	ret["TE"] = TE;
	return(ret);
}
