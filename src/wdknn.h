#pragma once

#include "types.h"

class WDKNNClassifier {
public:
    WDKNNClassifier(unsigned int n_neighbors);

    void fit(SparseMatrix X, Matrix y);

    Vector predict(SparseMatrix X);
private:
    unsigned int _n_neighbors;
    SparseMatrix _data;
    Matrix _labels;
    double peso_vecino(double distancia, int vecino);
};
