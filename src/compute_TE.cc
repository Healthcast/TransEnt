/**
* @file compute_TE.cc
* @brief This package is for computing Transfer Entropy estimation using three different methods
* @author
* @version
*/
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>

/* WARNING!!
 * The algorithms used here assume nearest neighbors are determined using the max norm. We implement the
 * nearest neigbhor search using the ANN (Approximate Nearest Neighbor) library. ANN is set to use the
 * max norm by adjusting some variables in the library's header files. Keep this in mind if you choose
 * to extend this library.
 */
#include <ANN/ANN.h>	// ANN declarations

#include <boost/tokenizer.hpp>							// for reading cvd
#include <boost/algorithm/string/trim.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include "boost/program_options.hpp"  					// for parsing command line arguments

using namespace std;
using namespace boost;
using namespace math;

bool DEBUG  = 0; 			// debug mode

const size_t ERROR_IN_COMMAND_LINE = 1;
const size_t SUCCESS = 0;
const size_t ERROR_UNHANDLED_EXCEPTION = 2;


FILE *pFilexky, *pFilex, *pFileky, *pFilexk, *pFilek;

int safetyCheck(const vector<double>&X,const vector<double>&Y,int embedding,
	       ANNpointArray &xkyPts,ANNpointArray &kyPts,
	       ANNpointArray &xkPts,ANNpointArray &kPts,
		   int nPts)
{
	for(int i=0;i<nPts;i++)
	{
		ANNpoint tmpPxky = xkyPts[i];
		ANNpoint tmpPky  = kyPts[i];
		ANNpoint tmpPxk  = xkPts[i];
		ANNpoint tmpPk   = kPts[i];
		for(int j=i+1;j<nPts;j++)
		{
			if(tmpPxky == xkyPts[j])
			{
				throw invalid_argument("Points with same coordinates in the XKY tree (add noise).");
				return -1;
			}
			if(tmpPky == kyPts[j])
			{
				throw invalid_argument("Points with same coordinates in the KY tree (add noise).");
				return -1;
			}
			if(tmpPxk == xkPts[j])
			{
				throw invalid_argument("Points with same coordinates in the XK tree (add noise).");
				return -1;
			}
			if(tmpPk == kPts[j])
			{
				throw invalid_argument("Points with same coordinates in the K tree (add noise).");
				return -1;
			}
		}

	}
	for(int i=embedding;i<X.size();i++){
		for(int j=i+1;j<X.size();j++){
			if(X[i]==X[j])
			{
				throw invalid_argument("Points with same coordinates in the X tree (add noise).");
				return -1;
			}
		}
	}
	return 1;
}

// make spaces xky, ky, xk, k
int MakeSpaces(const vector<double>&X,const vector<double>&Y,int embedding,bool safetyChk,
	       ANNpointArray &xkyPts,ANNpointArray &kyPts,
	       ANNpointArray &xkPts,ANNpointArray &kPts,
	       ANNkd_tree* &xkykdTree,ANNkd_tree* &kykdTree,
	       ANNkd_tree* &xkkdTree,ANNkd_tree* &kkdTree)
{
  if(DEBUG)
  {
     pFilexky = fopen("xky.space","w");
     pFilex   = fopen("x.space","w");
     pFileky  = fopen("ky.space","w");
     pFilexk  = fopen("xk.space","w");
     pFilek   = fopen("k.space","w");
  }
  int maxPts =  X.size()+1;						   //max number of points
  int nPts   =  0;
  int dimxky =  embedding + 2,
      dimky	 =  embedding + 1,
      dimxk	 =  embedding + 1,
      dimk   =  embedding;

  xkyPts = annAllocPts(maxPts, dimxky);				// allocate data points
  kyPts  = annAllocPts(maxPts, dimky);				// allocate data points
  xkPts  = annAllocPts(maxPts, dimxk);				// allocate data points
  kPts   = annAllocPts(maxPts, dimk);				// allocate data points

  for(int i=embedding;i<X.size();i++){
	if(i>Y.size())	break;
    int t=0;
    if(DEBUG)	fprintf(pFilex,"%f\n",X[i]);
    xkyPts[nPts][t]=X[i];if(DEBUG)	fprintf(pFilexky,"%f,",X[i]);
    xkPts[nPts][t]=X[i];if(DEBUG)	fprintf(pFilexk,"%f,",X[i]);
    t++;
    for(int j=1;j<=embedding;j++){//k
      xkyPts[nPts][t]=X[i-j];if(DEBUG)	fprintf(pFilexky,"%f,",X[i-j]);
      xkPts[nPts][t]=X[i-j];if(DEBUG)	fprintf(pFilexk,"%f,",X[i-j]);
      kyPts[nPts][t-1]=X[i-j];if(DEBUG)	fprintf(pFileky,"%f,",X[i-j]);
      kPts[nPts][t-1]=X[i-j];if(DEBUG)	fprintf(pFilek,"%f,",X[i-j]);
      t++;
    }
    xkyPts[nPts][t]=Y[i-1];if(DEBUG)	fprintf(pFilexky,"%f,",Y[i-1]);
    kyPts[nPts][t-1]=Y[i-1];if(DEBUG)	fprintf(pFileky,"%f,",Y[i-1]);
    nPts++;
    if(DEBUG)	fprintf(pFilexky,"\n");
    if(DEBUG)	fprintf(pFilexk,"\n");
    if(DEBUG)	fprintf(pFileky,"\n");
    if(DEBUG)	fprintf(pFilek,"\n");
  }
  if(safetyChk)
	  safetyCheck(X,Y,embedding,xkyPts,kyPts,xkPts,kPts,nPts);
  xkykdTree = new ANNkd_tree(xkyPts, nPts, dimxky);
  for(int i=0;i<nPts;i++)
  {

  }
  kykdTree  = new ANNkd_tree(kyPts, nPts, dimky);
  xkkdTree  = new ANNkd_tree(xkPts, nPts, dimxk);
  kkdTree   = new ANNkd_tree(kPts, nPts, dimk);
  return nPts;
}

double findDistanceK(ANNkd_tree* kdTree,int k,ANNpoint	 Pt)
{
  ANNidxArray		nnIdx;								// near neighbor indices
  ANNdistArray		dists;								// near neighbor distances
  double			distanceK;
  nnIdx = new ANNidx[k];								// allocate near neigh indices
  dists = new ANNdist[k];								// allocate near neighbor dists
  // search
  kdTree->annkSearch(Pt,							// query point
		     	 	 k,									// number of near neighbors
					 nnIdx,								// nearest neighbors (returned)
					 dists,								// distance (returned)
					 0);								// error bound

  distanceK = dists[k-1];								// distance to Kth neighbor
  delete [] nnIdx;										// clean things up
  delete [] dists;
  return distanceK;
}

ANNidx kthNeighbor(ANNkd_tree* kdTree,int k,ANNpoint	 Pt)
{
  ANNidxArray		nnIdx;								// near neighbor indices
  ANNdistArray		dists;								// near neighbor distances
  nnIdx = new ANNidx[k];								// allocate near neigh indices
  dists = new ANNdist[k];								// allocate near neighbor dists
  // search
  kdTree->annkSearch(Pt,							// query point
		     	 	 k,									// number of near neighbors
					 nnIdx,								// nearest neighbors (returned)
					 dists,								// distance (returned)
					 0);								// error bound

  ANNidx idx = nnIdx[k-1];
  delete [] nnIdx;										// clean things up
  delete [] dists;
  return idx;
}


// return index of the kth neighbor to Pt in kdtree (this version is not necessary because ANN guarantee order of neighbors)
ANNidx kthNeighbor_graunteed(ANNkd_tree* kdTree, int k, ANNpoint Pt){
  ANNidxArray    nnIdx = new ANNidx[k];    // allocate near neigh indices
  ANNdistArray    dists = new ANNdist[k];    // allocate near neighbor dists
  kdTree->annkSearch(Pt,k,nnIdx,dists,0);
  double max=0; int maxi=-1;
  for(int i=0;i<k;i++){ if(dists[i]>max){max=dists[i];maxi=i;} }
  ANNidx ans=nnIdx[maxi];
  delete [] nnIdx;
  delete [] dists;
  return(ans);
}


int countByDistanceView(ANNkd_tree* kdTree, ANNpoint Pt, double Distance)
{
  int cnt= kdTree->annkFRSearch(Pt,    // query point
                Distance,    // the distance within which the neighbors are counted
                0,        // since the count is needed, these 3 parameters can be null
                NULL,
                NULL,
                0);
  if(DEBUG)	printf("dist: %f cnt: %d\n",Distance,cnt);
  ANNidxArray    nnIdx = new ANNidx[cnt];    // allocate near neigh indices
  ANNdistArray    dists = new ANNdist[cnt];    // allocate near neighbor dists
  cnt= kdTree->annkFRSearch(Pt,
                Distance,    // the distance within which the neighbors are counted
                cnt,
                nnIdx,
                dists,
                0);
  if(DEBUG)
  for(int i=0;i<cnt;i++){
	  printf("%d of %d: %f dist %f pindx %d\n",i,cnt,abs(dists[i]-Distance),dists[i],nnIdx[i]);
  }
  if(DEBUG)	printf("-----------\n");

  return cnt;
}

// The count is the number of points less than or equal to the distance
int countByDistance(ANNkd_tree* kdTree, ANNpoint Pt, double Distance)
{
  int cnt= kdTree->annkFRSearch(Pt,				// query point
								Distance,		// the distance within which the neighbors are counted (square distance)
								0,						// since the count is needed, these 3 parameters can be null
								NULL,
								NULL,
								0);					// error bound

  if(DEBUG){
	  printf("cnt %d\n",cnt);
	  int fooCnt = countByDistanceView(kdTree, Pt, Distance);
	  if(cnt != fooCnt)
		  printf("cnt not match\n");
  }

  return cnt;
}


double compute_avg_distance_MI_diff(int nPts, int k,
									ANNkd_tree* xkykdTree,
									ANNpointArray	&xkyPts)
{
  double avDist=0;
  // For each point in the XKY  space,
  for(int i=0;i<nPts;i++){
    // Find the distance to the query point's kth neighbor in the XKY space
    avDist += findDistanceK(xkykdTree, k, xkyPts[i]);;
  }
  avDist/= nPts;
  return avDist;
}


/**
 * @brief 		Calculate Transfer Entropy estimation based on difference between mutual information
 * @details 	The transfer entropy can be expressesed as the difference between two mutual information measures
 * 					- the MI between the next point and its lagged values PLUS the second space,
 *   	   	   	   	- the MI between the next point and its lagged values.
 *   	   	   	We estimate the MI using Kraskov et al's method (2004), which is based on k-nearest neighbor distances.
 *   	   	   	In essence, this is an adaptive kernel density estimate, where the kernel is adaptively resized to the
 *   	   	   	amount of information on the underlying probability densities at each observation.
 *
 * @param[in]	nPts number of points
 * @param[in]   k The k'th neighbor
 * @param[in]   embedding The embedding dimension
 * @param[in]   xkykdTree kdtree of xky space
 * @param[in]   kykdTree  kdtree of  ky space
 * @param[in]   xkkdTree  kdtree of xk  space
 * @param[in]   kkdTree   kdtree of  k  space
 * @param[in]   X point of x space (1 dimensional)
 * @param[in]   xkyPts points of xky space
 * @param[in]   kyPts  points of  ky space
 * @param[in]   xkPts  points of xk  space
 * @param[in]   kPts   points of  k  space
 * @return		transfer entropy
 */
double TE_mutual_information_difference(int nPts, int k, int embedding,
                    ANNkd_tree* xkykdTree,
                    ANNkd_tree* kykdTree,
                    ANNkd_tree* xkkdTree,
                    ANNkd_tree* kkdTree,
                    const vector<double>&X,
                    ANNpointArray    &xkyPts,
                    ANNpointArray    &kyPts,
                    ANNpointArray    &xkPts,
                    ANNpointArray    &kPts)
{

// dimensions of the different spaces
  int dimxky =  embedding + 2,
    dimky    =  embedding + 1,
    dimxk    =  embedding + 1,
    dimk     =  embedding;

  // variables to store the distance to the kth neighbor in different spaces
  double tmpdist,xdistXKY,xdistXK,kydist,kdist;
  // counters for summing the digammas of the point counts.
  double cntX=0, cntX2=0, cntKY=0, cntK=0;
  // temporary counters
  int  fooCnt, barCnt;
  double avDist=0,avD2=0; //DEBUG
  // For each point in the XKY  space,
  for(int i=0;i<nPts;i++){
    // Find index of query point's kth neighbor in the XKY space
    ANNidx idx=kthNeighbor(xkykdTree, k, xkyPts[i]);
    // compute X distance and KY distance
    // ASSUMES points are x, k1,k2,..kn, y
    xdistXKY= abs(xkyPts[i][0] - xkyPts[idx][0]);
    kydist= abs(xkyPts[i][1] - xkyPts[idx][1]);
    for(int j=2;j<dimxky;j++){
      tmpdist=abs(xkyPts[i][j] - xkyPts[idx][j]);
      if(tmpdist>kydist){ kydist=tmpdist; }
    }
    // and in the XK space
    idx=kthNeighbor(xkkdTree, k, xkPts[i]);
    xdistXK=abs(xkPts[i][0] - xkPts[idx][0]);
    kdist=abs(xkPts[i][1] - xkPts[idx][1]);
    for(int j=2;j<dimxk;j++){
      tmpdist=abs(xkPts[i][j] - xkPts[idx][j]);
      if(tmpdist>kdist){ kdist=tmpdist; }
    }
    if(xdistXKY==0){
	  printf("x (XKY) crashing at %d, %d \n\t\n",i,idx);
	  for(int j=0;j<dimxky;j++){ printf("%f\t",xkyPts[i][j]); }
	  printf("\n\t\n");
	  for(int j=0;j<dimxky;j++){ printf("%f\t",xkyPts[idx][j]); }
	  printf("\n\t\n");
	  throw invalid_argument("There is a problem in the data. Please run the program with safety check.");
	  return -1;
	}
	if(xdistXK==0){
	  printf("x (XK) crashing at %d, %d \n\t\n",i,idx);
	  for(int j=0;j<dimxk;j++){ printf("%f\t",xkPts[i][j]); }
	  printf("\n\t\n");
	  for(int j=0;j<dimxk;j++){ printf("%f\t",xkPts[idx][j]); }
	  printf("\n\t\n");
	  throw invalid_argument("There is a problem in the data. Please run the program with safety check.");
	  return -1;
	}
	if(kydist==0){
	  printf("ky crashing at %d, %d \n\t\n",i,idx);
	  for(int j=0;j<dimxky;j++){ printf("%f\t",xkyPts[i][j]); }
	  printf("\n\t\n");
	  for(int j=0;j<dimxky;j++){ printf("%f\t",xkyPts[idx][j]); }
	  printf("\n\t\n");
	  throw invalid_argument("There is a problem in the data. Please run the program with safety check.");
	  return -1;
	}
	if(kdist==0){
	  printf("k crashing at %d, %d \n\t\n",i,idx);
	  for(int j=0;j<dimxk;j++){ printf("%f\t",xkPts[i][j]); }
	  printf("\n\t\n");
	  for(int j=0;j<dimxk;j++){ printf("%f\t",xkPts[idx][j]); }
	  printf("\n\t\n");
	  throw invalid_argument("There is a problem in the data. Please run the program with safety check.");
	  return -1;
	}
    // Count the number of points in X subspace within these distances
    // since this is a 1-d space, ASSUMING faster by a loop than by
    // kd-tree lookups
    fooCnt=0; barCnt=0;
    for(int j=embedding;j<X.size();j++){
      // NOTE! less than or equal to (<=)
      if( (abs(xkyPts[i][0] - X[j]) <= xdistXKY) && (abs(xkyPts[i][0] - X[j])!=0) ) fooCnt++;
      if( (abs(xkPts[i][0]  - X[j]) <= xdistXK ) && (abs(xkPts[i][0]  - X[j])!=0) ) barCnt++;
    }
    //avDist += fooCnt; avD2 += barCnt; // DEBUG
    cntX  += digamma(fooCnt); // and sum the digamma of the counts
    cntX2 += digamma(barCnt);
    // Count the number of points in the KY subspace, using the XKY distance:
    // and re-using fooCnt
    fooCnt = countByDistance(kykdTree, kyPts[i], kydist);
    cntKY += digamma(fooCnt); // and sum its digamma
    //avDist += fooCnt;  // DEBUG
    // and in the K subspace, using the XK distance:
    fooCnt = countByDistance(kkdTree,  kPts[i],  kdist);
    cntK += digamma(fooCnt);
    //avD2 += fooCnt; // DEBUG
    // if(i>2) break; // DEBUG
    if(DEBUG) printf("npts %d\n",i);
  }
  if(DEBUG) printf("av dist: %f\n",avDist/nPts);
  // The transfer entropy is the difference of the two mutual informations
  // If we define  digK = digamma(k),  digN = digamma(nPts); then the
  // Kraskov (2004) estimator for MI gives
  // TE = (digK - 1/k - (cntX + cntKY)/nPts + digN) - (digK - 1/k - (cntX2 + cntK)/nPts + digN)
  // which simplifies to:
  double TE = (cntX2 + cntK)/nPts - (cntX + cntKY)/nPts;
  return TE;
}


/**
 * @brief 		Calculate Transfer Entropy estimation based on ???
 * @details 	???
 *
 * @param[in]	nPts number of points
 * @param[in]   k The k'th neighbor
 * @param[in]   embedding The embedding dimension
 * @param[in]   xkykdTree kdtree of xky space
 * @param[in]   kykdTree  kdtree of  ky space
 * @param[in]   xkkdTree  kdtree of xk  space
 * @param[in]   kkdTree   kdtree of  k  space
 * @param[in]   xkyPts points of xky space
 * @param[in]   kyPts  points of  ky space
 * @param[in]   xkPts  points of xk  space
 * @param[in]   kPts   points of  k  space
 * @return		transfer entropy
 */
double TE_direct(int nPts, int k, int embedding,
				 ANNkd_tree* xkykdTree,
				 ANNkd_tree* kykdTree,
				 ANNkd_tree* xkkdTree,
				 ANNkd_tree* kkdTree,
				 ANNpointArray		&xkyPts,
				 ANNpointArray		&kyPts,
				 ANNpointArray		&xkPts,
				 ANNpointArray		&kPts)
{
  int dimxky =  embedding+2,										// dimensions of different spaces
	  dimky  =  embedding+1,
	  dimxk  =  embedding+1,
	  dimk   =  embedding;

  int		cntKY,cntXK,cntK;										// counters for the count of points in each subspace
  double 	Total=0;												// for summing up digamma of counters

  for(int i=0;i<nPts;i++){
    double dist = findDistanceK(xkykdTree, k, xkyPts[i]);	// distance to kth neighbor in xky space
    cntKY = countByDistance(kykdTree, kyPts[i], dist);		// count of points within the distance in ky subspace
    cntXK = countByDistance(xkkdTree, xkPts[i], dist);		// count of points within the distance in xk subspace
    cntK  = countByDistance(kkdTree,  kPts[i],  dist);		// count of points within the distance in k  subspace
    Total += (digamma(cntK)-digamma(cntXK)-digamma(cntKY));	// sum of digamma of counts
  }
  // calculating the transfer entropy
  double TE = digamma(k)-(1/k)+ (Total/nPts); //TODO: m-1/k or 1/k?
  return TE;
}

/**
 * @brief 		Calculate Transfer Entropy estimation based on ???
 * @details 	???
 *
 * @param[in]	nPts number of points
 * @param[in]   k The k'th neighbor
 * @param[in]   embedding The embedding dimension
 * @param[in]   xkykdTree kdtree of xky space
 * @param[in]   kykdTree  kdtree of  ky space
 * @param[in]   xkkdTree  kdtree of xk  space
 * @param[in]   kkdTree   kdtree of  k  space
 * @param[in]   xkyPts points of xky space
 * @param[in]   kyPts  points of  ky space
 * @param[in]   xkPts  points of xk  space
 * @param[in]   kPts   points of  k  space
 * @param[in]   eDistance Distance used for measuring TE in Correlation method, by default it is the average distance calculated in XKY space
 * @return		transfer entropy
 */
double TE_generalize_correlation_sum(int nPts, int k, int embedding,
				     	 	 	 	 ANNkd_tree* xkykdTree,
									 ANNkd_tree* kykdTree,
									 ANNkd_tree* xkkdTree,
									 ANNkd_tree* kkdTree,
									 ANNpointArray		&xkyPts,
									 ANNpointArray		&kyPts,
									 ANNpointArray		&xkPts,
									 ANNpointArray		&kPts,
									 double eDistance)
{
  if(DEBUG) printf("dist: %f\n",eDistance);
  int dimxky =  embedding+2,											// dimensions of different spaces
   	  dimky  =  embedding+1,
	  dimxk  =  embedding+1,
	  dimk   =  embedding;

  double  cntXKY, cntK, cntXK, cntKY,									// counters for the count of points in each subspace
		  Total=0;														// for summing up log of cnts
int foo=0;
  for(int i=0;i<nPts;i++){
	if(DEBUG)	printf("dist %f\n",eDistance);
	cntXKY = countByDistance(xkykdTree, xkyPts[i], eDistance);// counts of points in XKY space within eDistance from point i
	cntKY  = countByDistance( kykdTree,  kyPts[i], eDistance);// counts of points in KY space within eDistance from point i
	cntXK  = countByDistance( xkkdTree,  xkPts[i], eDistance);// counts of points in XK space within eDistance from point i
	cntK   = countByDistance(  kkdTree,   kPts[i], eDistance);// counts of points in K space within eDistance from point i

	// if cntXKY is zero, counts in other cnts have to be zero as well
	if(cntXKY != 0)														// error checking
	{
	  if(cntKY == 0)	printf("cntKY is Zero!!!\n");
	  if(cntXK == 0)	printf("cntXK is Zero!!!\n");
	  if(cntK == 0)		printf("cntK is Zero!!!\n");
	}
	// if cntXKY is zero the log(0) is undefined so do not add anything
	if(cntXKY != 0)
	  Total+= log2((cntXKY*cntK)/(cntXK*cntKY));
	if(cntXKY==0) foo++;
  }
  if(DEBUG) printf("%d zeros\n",foo);
  // calculating TE:
  double TE = TE = Total/nPts;
  return TE;
}

/**
 * @brief 		Compute estimation of Transfer Entropy between two random process
 * @details 	Compute_TE is a function to compute Transfer Entropy given a method
 * 				(Direct, MI_diff [Mutual Information difference] and, correlation)
 *	 			TODO: should I add more detail?
 *
 * @param[out]	TE The estimated transfer entropy
 * @param[in]	X Transfer Entropy is calculated to random process X (with noise)
 * @param[in]   Y Transfer Entropy is calculated from random process Y (with noise)
 * @param[in]   embedding The embedding dimension
 * @param[in]   k The k'th neighbor
 * @param[in]   method The method to be used to estimate TE
 * @param[in]   epsDistace Distance used for measuring TE in Correlation method, by default it is the average distance calculated in XKY
 * @param[in]   safetyCheck For computing TE using "mi_diff" method the data need to be noisy otherwise a crach might happen. This parameter can check if there are any idetical points in the spaces made for this use.
 * @return		SUCCESS/ERORR code
 */
int compute_TE(double& TE, vector<double>&X, vector<double>&Y, int embedding, int k, string method, double epsDistance=-1, bool safetyChk=false){
  if(DEBUG)	printf("%f\n",epsDistance);
  if( method != "MI_diff" 	&& method != "mi_diff" &&
/*    method != "Direct"  	&& method != "direct"  */
      method != "Correlation" 	&& method != "correlation")
	{ // TODO! Correct this message
	  throw invalid_argument("Method not specified correctly. Please choose one of the following. (\"MI_diff\", \"Correlation\")");
	  return -1;//error
	}

  ANNpointArray		xkyPts;				// data points
  ANNpointArray		kyPts;				// data points
  ANNpointArray		xkPts;				// data points
  ANNpointArray		kPts;				// data points
  ANNkd_tree*		xkykdTree;			// search structure
  ANNkd_tree*		kykdTree;			// search structure
  ANNkd_tree*		xkkdTree;			// search structure
  ANNkd_tree*		kkdTree;			// search structure
  int dimxky =  embedding + 2, // embedding dimensions
      dimky  =  embedding + 1,
      dimxk  =  embedding + 1,
      dimk   =  embedding,
      nPts;								//number of points
  // making all spaces (xky, ky, xk, k) and kdtrees
  nPts = MakeSpaces(X,Y,embedding,safetyChk,xkyPts,kyPts,xkPts,kPts,xkykdTree,kykdTree,xkkdTree,kkdTree);
  // choosing the method for calculating TE
  if(method == "mi_diff" || method == "MI_diff")
  {
    TE = TE_mutual_information_difference(nPts, k, embedding, xkykdTree, kykdTree, xkkdTree, kkdTree, X, xkyPts, kyPts, xkPts, kPts);
  }
/*  else if(method == "Direct")
  {
    TE =    					TE_direct(nPts, k, embedding, xkykdTree, kykdTree, xkkdTree, kkdTree,    xkyPts, kyPts, xkPts, kPts);
  }*/
  else if(method == "Correlation" || method == "correlation")
  {
	if (epsDistance == -1)
		epsDistance = compute_avg_distance_MI_diff(nPts, k, xkykdTree, xkyPts);
	if(DEBUG) printf("avg distance: %f\n",epsDistance);
    TE =    TE_generalize_correlation_sum(nPts, k, embedding, xkykdTree, kykdTree, xkkdTree, kkdTree,    xkyPts, kyPts, xkPts, kPts, epsDistance);
  }

  // clean up
  delete xkkdTree;
  delete kkdTree;
  delete xkykdTree;
  delete kykdTree;
  annDeallocPts(xkyPts);
  annDeallocPts(kyPts);
  annDeallocPts(kPts);
  annDeallocPts(xkPts);
  annClose();
  return SUCCESS;
}

void makeXY(string data,vector<double>&X,vector<double>&Y)
{
	ifstream input(data.c_str());
	// Read the data from the csv file, X and Y should be the first and second columns of the file
	typedef tokenizer< escaped_list_separator<char> > Tokenizer;
	vector< string > vec;
	string line;
	while (getline(input,line))
	{
		Tokenizer tok(line);
		vec.assign(tok.begin(),tok.end());
		trim(vec[0]);
		trim(vec[1]);
		X.push_back(atof(vec[0].c_str()));
		Y.push_back(atof(vec[1].c_str()));
	}
	X.erase(X.begin()); // Using boost to read csv X[0] and Y[0] are null
	Y.erase(Y.begin());
	return;
}

/*
int main(int argc, char** argv)
{
  bool VERBOSE= 0;
  int lookBack = 3;
  int k = 1;
  string data("test4_1K.csv");
  string method("MI_diff");
  double corDist= -1;
  try
  {
	namespace po = boost::program_options;
	po::options_description desc("Options");
	desc.add_options()
		("help,h", 												    "Print help messages")
		("verbose,v",											    "print words with verbosity")
		("embedding,e",   po::value<int>(&lookBack)->required(),    "Lookback size")
		("kthNeighbor,k", po::value<int>(&k)                 ,      "K'th neighbor")
	    ("input,i",       po::value<string>(&data)->required(),     "Data input file")
		("method,m",      po::value<string>(&method),               "Method for calculating TE (\"MI_diff\", \"Correlation\")")
		("correlationDistance,d",      po::value<double>(&corDist), "Distance for calculation TE with Correlation method");
	po::variables_map vm;
	try
	{
      po::store(po::parse_command_line(argc, argv, desc), vm);
	  if ( vm.count("help")  )
	  {
		cout<<"Usage: "<<argv[0]<<" -e <Lookback> -i <DataInputFile> [-v] [-m <method[\"MI_diff\", \"Correlation\"]>] [-d <distance for correlation method>] [-k <k'th neighbor>] [-n]"<<endl<<endl;
		return SUCCESS;
	  }
	  if ( vm.count("verbose")  )
	  {
		VERBOSE = 1;
	  }
	  po::notify(vm);
	}
	catch(boost::program_options::required_option& e)
	{
	  std::cerr << "ERROR: " << e.what() << endl << endl;
	  cout<<"Usage: "<<argv[0]<<" -e <Lookback> -i <DataInputFile> [-v] [-m <method[\"MI_diff\", \"Correlation\"]>] [-d <distance for correlation method>] [-k <k'th neighbor>] [-n]"<<endl<<endl;
	  return ERROR_IN_COMMAND_LINE;
	}
  }
  catch(std::exception& e)
  {
    std::cerr << "Unhandled Exception reached the top of main: "
              << e.what() << ", application will now exit" << std::endl;
    return ERROR_UNHANDLED_EXCEPTION;
  }

  if (VERBOSE)  cout<<"Lookback "<<lookBack<< "\tdatafile " << data<<endl;
  double TE;

// making the result for 11 different coefficients
//  FILE *foo;
//  foo = fopen(plotfile.c_str(),"w");
//  fprintf(foo,"c,te\n");
//  if (method == "Correlation")
//  {
//	  string files[11] = {"test0-01.csv", "test0-1.csv", "test0-2.csv", "test0-3.csv", "test0-4.csv", "test0-5.csv",
//					      "test0-6.csv" , "test0-7.csv", "test0-8.csv", "test0-9.csv", "test1-0.csv"};
//	  for(int i=0;i<11;i++)
//	  {
//		  vector<double>X;
//		  vector<double>Y;
//		  makeXY("Data/"+files[i],X,Y);
//		  compute_TE(TE, X, Y, lookBack, k, "Correlation",plotfile,exp(-1));
//		  fprintf(foo,"%s,%f\n",files[i].c_str(),TE);
//	  }
//  }

  try{
	  vector<double>X;
	  vector<double>Y;
	  makeXY(data,X,Y);
	  compute_TE(TE, X, Y, lookBack, k, method,corDist);
	  if(VERBOSE){ cout<<method<<" X,Y: "<<TE<<endl; }
	  //compute_TE(TE, Y, X, lookBack, k, method,corDist);
	  //if(VERBOSE){ cout<<method<<" Y,X: "<<TE<<endl; }
  }
  catch(invalid_argument& e)
  {
	  cerr << e.what() << endl;
	  return -1;
  }
// compute_TE(TE, X, Y, lookBack, "Correlation",0.55);
// if(VERBOSE){ cout<<"Correlation X,Y: "<<TE<<endl; }
// compute_TE(TE, X, Y, lookBack, "Correlation",0.65);
// if(VERBOSE){ cout<<"Correlation X,Y: "<<TE<<endl; }
//  compute_TE(TE, X, Y, lookBack, "Correlation",0.75);
//  if(VERBOSE){ cout<<"Correlation X,Y: "<<TE<<endl; }
//  compute_TE(TE, X, Y, lookBack, "Correlation",1);
//  if(VERBOSE){ cout<<"Correlation X,Y: "<<TE<<endl; }
//  compute_TE(TE, X, Y, lookBack, "Direct");
//  if(VERBOSE){ cout<<"Direct X,Y: "<<TE<<endl; }
//  if(VERBOSE){ cout<<"-------------------"<<endl; }

  return SUCCESS;

}
*/
