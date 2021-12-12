// Copied from inverter_wrapper.i

// Tell swig the name of the module we're creating
%module lut_wrapper

// Pull in the headers from Python itself and from our library
%{
#define SWIG_FILE_WITH_INIT
#include <Python.h>
#include "lut_amm.hpp"
%}

// typemaps.i is a built-in swig interface that lets us map c++ types to other
// types in our language of choice. We'll use it to map Eigen matrices to
// Numpy arrays.
%include <typemaps.i>
%include <std_vector.i>

// eigen.i is found in ../swig/ and contains specific definitions to convert
// Eigen matrices into Numpy arrays.
%include <eigen.i>

%eigen_typemaps(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>)
%eigen_typemaps(Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>)
%eigen_typemaps(Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>)
%eigen_typemaps(Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>)


// Tell swig to build bindings for everything in our library
%include "lut_amm.hpp"
