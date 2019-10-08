#pragma once

#include "types.h"


class KNNClassifier {
public:
    KNNClassifier(unsigned int n_neighbors);

    void fit(SparseMatrix X, Matrix y);

    Vector predict(SparseMatrix X);
    Vector predict_weighted(SparseMatrix X,const Vector& correlaciones);
    double weighted_norm(const Vector& v,const Vector& pesos);
private:
    unsigned int _n_neighbors;
    Matrix _data;
    Matrix _labels;
};
