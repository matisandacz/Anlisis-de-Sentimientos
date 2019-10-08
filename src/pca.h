#pragma once
#include "types.h"

class PCA {
public:
    PCA(unsigned int n_components);

    void fit(Matrix X);

    Eigen::MatrixXd transform(SparseMatrix X, int alpha);
private:
    unsigned _n_components; //cantidad maxima de componentes que puede dar
    Matrix _transf;
};
