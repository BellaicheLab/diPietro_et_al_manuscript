#ifndef __trackingincludes_h__
#define __trackingincludes_h__

// ----- ITK includes -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageRegionIterator.h>
#include <itkConnectedComponentImageFilter.h>
#include <itkConstNeighborhoodIterator.h>
#include "itkLabelGeometryImageFilter.h"
#include "itkPermuteAxesImageFilter.h"


// ----- STD includes -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#include <cstdio>
#include <fstream>
#include <map>
#include <vector>
#include <string>
#include <ctime>        // for benchmark
#include <cctype>       // for isdigit
#include <algorithm>    // for sort,find,... on STL containers 
#include <numeric>
#include <sstream>
#include <iostream>

// ----- Custom includes --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#include "CorrespondanceCouple.h"
#include "Couple.h"
#include "tools.h"  //(3.0) toolbox

// Current version number
const std::string TRACKING_VERSION = "4.0.1";

// ----- Golbal typedefs definition ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
/*
* Allows to redefine particular variable types under a new and more adapted name. Highly adviced when using templated function such as ITK functions
* @TODO: They should not be declared as global ...
*/
typedef unsigned long PixelType;                                                         // Pixel Type
typedef std::pair<std::string, std::string> ANPairType;                                  // Absolute Number Pair Type
typedef std::pair<PixelType, PixelType> RNPairType;                                      // Relative Number Pair Type
typedef itk::Image<PixelType, 2> ITKImageType;                                           // ITK Image Type 
typedef itk::ImageFileReader<ITKImageType> ReaderType;                                   // ITK Reader Type 
typedef itk::ConnectedComponentImageFilter<ITKImageType, ITKImageType> LabelFilterType;  // ITK Label Filter Type 
typedef itk::LabelGeometryImageFilter<ITKImageType, ITKImageType> GeometryFilterType;    // ITK Geometry Filter Type 
typedef itk::ImageRegionIterator<ITKImageType> IteratorType;                             // ITK Region Iterator Type 
typedef itk::ImageRegionConstIterator<ITKImageType> ConstIteratorType;                   // ITK Const Region iterator Type 
typedef itk::ConstNeighborhoodIterator<ITKImageType> ConstNeighborhoodIteratorType;      // ITK Const Neighbourhood iterator Type  
typedef itk::PermuteAxesImageFilter<ITKImageType> TransposeImageFilterType;

// ----- Templated Function ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
/*
 * Must be define in .h file because of the template
 */

/* Allow the search of a AN pair in vector of pair
* return true if one of the element of the pair is equal to the seached value
*/
struct IsEqualStr
{
    IsEqualStr(std::string val) : val_(val) {}
    bool operator()(const std::pair<std::string, std::string>& elem) const 
    {
        return (val_.compare(elem.first) == 0 || val_.compare(elem.second) == 0);
    }
private:
    std::string val_;
};

/* Allow the search of a RN pair in vector of pair
* return true if one of the element of the pair is equal to the seached value
*/
struct IsEqualLong
{
    IsEqualLong(unsigned long val) : val_(val) {}
    bool operator()(const std::pair<unsigned long, unsigned long>& elem) const 
    {
        return (val_ == elem.first || val_ == elem.second);
    }
private:
    unsigned long val_;
};

/* Invert a pair order
* @param P the pair to invert
* @return the inverted pair
*/
template<class T>
std::pair<T, T> SwitchPair(const std::pair<T, T> &P)
{
    return std::make_pair(P.second, P.first);
}

/* Read parameters from a specific file.
* @param filename the file name
* @param bordercells the destination vector
*/
template <class T_value>
int ReadParams(std::ifstream & file, std::string PARAM, T_value & val)
{
    std::string line, tmp;
    bool found = false;
    while (file.good() && !found)
    {
        getline(file, line); // read the line
        if (!line.empty())
        {
            // line in the std::stringstream
            std::stringstream ss(line);
            ss >> tmp;
            if (tmp.compare(PARAM) == 0) //found
            {
                ss >> val;
                found = true;
            }
        }
    }
    file.clear();             // clear flag (MANDATORY))
    file.seekg(0, std::ios::beg);  // back to the beginning
    if (!found)
    {
        return 0;
    }
    else
    {
        return 1;
    }
}

#endif // __trackingincludes_h__
