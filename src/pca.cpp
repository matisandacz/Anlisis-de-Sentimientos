#include <iostream>
#include "pca.h"
#include "eigen.h"

using namespace std;


PCA::PCA(unsigned int n_components)
{
    _n_components = n_components;
}

void PCA::fit(Matrix X)
{
    Vector medias(X.cols());
    for(int j = 0; j < X.cols(); j++){
      double suma = 0;
      for(int i = 0; i < X.rows(); i++){
        suma += X(i,j);
      }
      suma /= X.rows();
      medias[j] = suma;
    }
    for(int i = 0; i < X.rows(); i++){
      for(int j = 0; j < X.cols(); j++){
        X(i,j) -= medias[j];
      }
    }
    Matrix Cov = (X.transpose()*X)/(X.rows()-1);
    _transf = get_first_eigenvalues(Cov,_n_components,1000,0.01).second;
}


MatrixXd PCA::transform(SparseMatrix X, int alpha)
{
  return MatrixXd(X*(_transf.leftCols(alpha)));
}
