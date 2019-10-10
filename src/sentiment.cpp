#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "knn.h"
#include "wdknn.h"
#include "pca.h"
#include "eigen.h"

namespace py=pybind11;

// el primer argumento es el nombre...
PYBIND11_MODULE(sentiment, m) {
    py::class_<KNNClassifier>(m, "KNNClassifier")
        .def(py::init<unsigned int>())
        .def("fit", &KNNClassifier::fit)
        .def("predict", &KNNClassifier::predict)
        .def("testearK", &KNNClassifier::testearK)
        .def("testearK_weighted", &KNNClassifier::testearK_weighted)
        .def("predict_weighted", &KNNClassifier::predict_weighted);

    py::class_<WDKNNClassifier>(m, "WDKNNClassifier")
        .def(py::init<unsigned int>())
        .def("fit", &WDKNNClassifier::fit)
        .def("predict", &WDKNNClassifier::predict)
        .def("predict_weighted", &WDKNNClassifier::predict_weighted)
        .def("set_mayority", &WDKNNClassifier::set_mayority)
        .def("set_dudani", &WDKNNClassifier::set_dudani)
        .def("set_zabrel", &WDKNNClassifier::set_zabrel)
        .def("set_fibonacci", &WDKNNClassifier::set_fibonacci)
        .def("set_inverse_distance", &WDKNNClassifier::set_inverse_distance);

    py::class_<PCA>(m, "PCA")
        .def(py::init<unsigned int>())
        .def("fit", &PCA::fit)
        .def("transform", &PCA::transform);
    m.def(
        "power_iteration", &power_iteration,
        "Function that calculates eigenvector",
        py::arg("X"),
        py::arg("num_iter")=5000,
        py::arg("epsilon")=1e-16
    );
    m.def(
        "get_first_eigenvalues", &get_first_eigenvalues,
        "Function that calculates eigenvector",
        py::arg("X"),
        py::arg("num"),
        py::arg("num_iter")=5000,
        py::arg("epsilon")=1e-16
    );
}
