#pragma once

#include "types.h"
#include <vector>
#include <queue>
#include <cmath>

using namespace std;

class WDKNNClassifier {
public:
    WDKNNClassifier(unsigned int n_neighbors);

    void fit(SparseMatrix X, Matrix y);

    Vector predict(SparseMatrix X);
    Vector predict_weighted(SparseMatrix X,const Vector& correlaciones);
    double weighted_norm(const Vector& v,const Vector& pesos);

    // setear funcion de votacion
    void set_mayority();
    void set_inverse_distance();
    void set_dudani();
    void set_zabrel();
    void set_fibonacci();


private:
    unsigned int _n_neighbors;
    SparseMatrix _data;
    Matrix _labels;
    vector<double> num_fibo;
    int vote_rule;

    // funciones de votacion
    void mayority_vote(priority_queue<double>& vecinosLabelPos, priority_queue<double>& vecinosLabelNeg, double& pesos_positivo, double& pesos_negativo);
    void inverse_distance_vote(priority_queue<double>& vecinosLabelPos, priority_queue<double>& vecinosLabelNeg, double& pesos_positivo, double& pesos_negativo);
    void dudani_vote(priority_queue<double>& vecinosLabelPos, priority_queue<double>& vecinosLabelNeg, double& pesos_positivo, double& pesos_negativo);
    void zabrel_vote(priority_queue<double>& vecinosLabelPos, priority_queue<double>& vecinosLabelNeg, double& pesos_positivo, double& pesos_negativo);
    void fibonacci_vote(priority_queue<double>& vecinosLabelPos, priority_queue<double>& vecinosLabelNeg, double& pesos_positivo, double& pesos_negativo);
};
