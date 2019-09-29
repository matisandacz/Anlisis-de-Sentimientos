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
    Vector medias = X.colwise().norm();
    X = X.rowwise() - medias;
    X = X.T*X / (X.rows()-1);
    //completar
}


MatrixXd PCA::transform(SparseMatrix X)
{
  throw std::runtime_error("Sin implementar");
}
