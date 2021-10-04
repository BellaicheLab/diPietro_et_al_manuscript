#ifndef __UTILITY_H
#define __UTILITY_H


#include "itkImage.h"

#include <map>
#include <iostream>
#include <fstream>

#include "itkSymmetricEigenAnalysis.h"

const char * Convert(const std::string & s)
{
    char *pc = new char[s.size() + 1];
    std::strcpy(pc, s.c_str());
    return pc;
}

template <typename M, typename V>
void MapToVec(const  M & m, V & v) 
{
    for (typename M::const_iterator it = m.begin(); it != m.end(); ++it) 
    {
        v.push_back(it->second);
    }
}

unsigned long CantorPairing(unsigned long a, unsigned long b)
{
    if (a > b)
    {
        unsigned long tmp = b;
        b = a;
        a = tmp;
    }
    return 0.5 * (a + b) * (a + b + 1) + b;
}




itk::Index<2> ind2sub(unsigned int w, unsigned int h, unsigned long i)
{
    itk::Index<2> res;
    res[0] = i % w;
    res[1] = i / w;
    return res;
}

unsigned long sub2ind(unsigned int w, unsigned int h, itk::Index<2> i)
{
    return i[0] + w*i[1];
}

float EuclideenDistance(itk::Index<2> _a, itk::Index<2> _b)
{
    return sqrt(std::pow(_a[0] - _b[0], 2) + std::pow(_a[1] - _b[1], 2));
}

float EuclideenDistance(unsigned int _u, unsigned int _v, unsigned int _w, unsigned int _h)
{
    itk::Index<2> a = ind2sub(_w, _h, _u);
    itk::Index<2> b = ind2sub(_w, _h, _v);
    return sqrt(std::pow(a[0] - b[0], 2) + std::pow(a[1] - b[1], 2));
}

void TensorData(std::vector< float > & m, itk::FixedArray< float, 2 > & eigenvalues, itk::Matrix< float, 2, 2 > & eigenvectors)
{
    typedef vnl_matrix< float > InputMatrixType;
    typedef itk::FixedArray< float, 2 > EigenValuesArrayType;
    typedef itk::Matrix< float, 2, 2 > EigenVectorMatrixType;
    typedef itk::SymmetricEigenAnalysis< InputMatrixType, EigenValuesArrayType, EigenVectorMatrixType > SymmetricEigenAnalysisType;

    InputMatrixType tensor (m.data(), 2, 2);
    SymmetricEigenAnalysisType symmetricEigenSystem;
    symmetricEigenSystem.SetDimension(2);
    symmetricEigenSystem.ComputeEigenValuesAndVectors(tensor, eigenvalues, eigenvectors);
}






#endif // __UTILITY_H