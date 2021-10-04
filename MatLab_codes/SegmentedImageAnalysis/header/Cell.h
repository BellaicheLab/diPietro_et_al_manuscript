/**
* \class Cell
*
*
* \brief Class regrouping all information related to a cell in a tissu
*
* This class gathered all information related to a cell, from its position
* in the image, its area, or perimeter, but also its relationship with
* other component of a tissu, such as other cells (neighbors), junctions,
* and vertices.
*
* \author $Author: Stephane Rigaud $
*
* \version $Revision: 1.0 $
*
* \date $Date: 2018/04/04 $
*
* Contact: stephane.rigaud@curie.fr
*
*/


#ifndef __CELL_H
#define __CELL_H

#include <numeric>
#include <vector>
#include <set>
#include <iostream>
#include <fstream>

#define _USE_MATH_DEFINES
#include <math.h>

#include "itkImage.h"
#include "Junction.h"
#include "Vertex.h"

#include "Utility.h"


class Cell 
{

public:

    static int WIDTH;
    static int HEIGHT;

    enum CATEGORY 
    {
        CORECELL = 0,
        FIRSTLAYER = 1,
        BORDERCELL = 2,
    };

    typedef unsigned long    LabelType;
    typedef itk::Index < 2 > IndexType;
    typedef itk::FixedArray < float, 2 > PointType;


    Cell();
    Cell(LabelType &);
    Cell(LabelType &, CATEGORY);
    ~Cell();

    LabelType GetLabel();
    void SetLabel(LabelType &);

    int  GetCategory();
    void SetCategory(int);
    void SetCategory(CATEGORY);

    bool operator == (const Cell &) const;

    void AddNeighbor(LabelType &);
    void AddJunction(LabelType &);
    void AddVertex  (LabelType &);
    void AddPixel   (IndexType &);

    std::vector < LabelType > GetNeighborsList();
    std::vector < LabelType > GetJunctionsList();
    std::vector < LabelType > GetVerticesList();
    std::vector < IndexType > GetPixelsList();
    std::vector < IndexType > GetSmallDilPixelsList();
    std::vector < LabelType > GetIndicesPixelsList();
    std::vector < LabelType > GetJunctionsIndicesPixelsList();
    std::vector < LabelType > GetDilJunctionsIndicesPixelsList();

    void SetNeighborsList(std::vector < LabelType > &);
    void SetJunctionsList(std::vector < LabelType > &);
    void SetVerticesList (std::vector < LabelType > &);
    void SetPixelsList   (std::vector < IndexType > &);

    PointType GetCentroid();
    float GetArea();
    float GetPerimeter();
    float GetElongation();
    float GetEccentricity();
    float GetRoundness();
    float GetAnisotropy();
    float GetOrientation();
    float GetMajorAxisLength();
    float GetMinorAxisLength();
    float GetChordLength();
    float GetChordDisorder();
    double GetMaximumIntensity();
    double GetMinimumIntensity();
    double GetAverageIntensity();
    double GetVarianceIntensity();
    double GetContourIntensity();
    double GetBackgroundIntensity();
    float GetInertiaAnisotropy();
    float GetInertiaOrientation();
    float GetVertexOrientation();
    float GetVertexPolarity();
    double GetContourIntensity(unsigned int);
    double GetBackgroundIntensity(unsigned int);
    double GetContourIntensityDisorder(unsigned int);

    std::vector<double> GetPolarityMode(unsigned int);
    std::vector<float> GetInertiaTensor();
    std::vector<float> GetTextureTensor();
    std::vector<float> GetVertexTensor();

    void SetArea(float &);
    void SetPerimeter(float &);
    void SetElongation(float &);
    void SetEccentricity(float &);
    void SetRoundness(float &);
    void SetAnisotropy(float &);
    void SetOrientation(float &);
    void SetMajorAxisLength(float &);
    void SetMinorAxisLength(float &);
    void SetMaximumIntensity (double &);
    void SetMinimumIntensity (double &);
    void SetAverageIntensity (double &);
    void SetVarianceIntensity(double &);
    void SetCentroid(float &, float &);

    void ComputeIntensities(itk::Image < double, 2 > ::Pointer, itk::Image < double, 2 > ::Pointer);
    void ComputePolarityMode(itk::Image < double, 2 > ::Pointer, itk::FixedArray < unsigned int, 2 > &, bool);
    void ComputeCategory(std::map < LabelType, Cell > &);
    void ComputeInertia(float, itk::FixedArray < unsigned int, 2 > &);
    void ComputeTexture(std::map < LabelType, Cell > &, std::map < LabelType, Junction > &, float, itk::FixedArray < unsigned int, 2 > &);
    void ComputeVertices(std::map < LabelType, Vertex > &, itk::FixedArray < unsigned int, 2 > &);
    void ComputeChords(std::map < LabelType, Junction > &);
    void ComputeJunctionsPixelsList(std::map < LabelType, Junction > &);

private:

    LabelType m_Label;
    CATEGORY m_Category;

    std::vector < LabelType > m_NeighborsList;
    std::vector < LabelType > m_VerticesList;
    std::vector < LabelType > m_JunctionsList;
    std::vector < IndexType > m_PixelsList;
    std::vector < IndexType > m_SmallDilPixelsList;
    std::vector < IndexType > m_JunctionsPixelsList;

    PointType m_Centroid;
    float m_Area = 0;
    float m_Perimeter = 0;
    float m_Elongation = 0;
    float m_Eccentricity = 0;
    float m_Roundness = 0;
    float m_Anisotropy = 0;
    float m_Orientation = 0;
    float m_MajorAxisLength = 0;
    float m_MinorAxisLength = 0;
    float m_ChordLength = 0;
    float m_ChordDisorder = 0;
    double m_MaximumIntensity = 0;
    double m_MinimumIntensity = 0;
    double m_AverageIntensity = 0;
    double m_VarianceIntensity = 0;
    double m_ContourIntensity = 0;
    double m_BackgroundIntensity = 0;
    float m_InertiaAnisotropy = 0;
    float m_InertiaOrientation = 0;
    float m_VertexPolarity = 0;
    float m_VertexOrientation = 0;

    std::vector< float > m_InertiaTensor;
    std::vector< float > m_TextureTensor;
    std::vector< float > m_VertexTensor;
    std::vector< double > m_BackgroundIntensityList;
    std::vector< double > m_ContoursIntensityList;
    std::vector< double > m_ContoursIntensityDisordersList;

    std::vector< std::vector<double> > m_PolarityModesList;

}; // end of class Cell



int Cell::WIDTH = 1;
int Cell::HEIGHT = 1;

Cell::Cell()
{
    m_InertiaTensor = std::vector< float >(4);
    m_TextureTensor = std::vector< float >(4);
    m_VertexTensor = std::vector< float >(4);
}

Cell::Cell(unsigned long & _l) : m_Label(_l)
{
    m_InertiaTensor = std::vector< float >(4);
    m_TextureTensor = std::vector< float >(4);
    m_VertexTensor = std::vector< float >(4);
}

Cell::Cell(unsigned long & _l, Cell::CATEGORY _c) : m_Label(_l), m_Category(_c)
{
    m_InertiaTensor = std::vector< float >(4);
    m_TextureTensor = std::vector< float >(4);
    m_VertexTensor = std::vector< float >(4);
}

Cell::~Cell()
{}

unsigned long Cell::GetLabel()
{
    return m_Label;
}

void Cell::SetLabel(unsigned long & _l)
{
    m_Label = _l;
}

int Cell::GetCategory()
{
    return static_cast<int>(m_Category);
}

void Cell::SetCategory(int _c)
{
    m_Category = static_cast<Cell::CATEGORY>(_c);
}

void Cell::SetCategory(Cell::CATEGORY _c)
{
    m_Category = _c;
}

bool Cell::operator==(const Cell & _p) const
{
    return this->m_Label == _p.m_Label;
}


std::vector<unsigned long> Cell::GetNeighborsList()
{
    return m_NeighborsList;
}

std::vector<unsigned long> Cell::GetJunctionsList()
{
    return m_JunctionsList;
}

std::vector<unsigned long> Cell::GetVerticesList()
{
    return m_VerticesList;
}

std::vector< itk::Index<2> > Cell::GetPixelsList()
{
    return m_PixelsList;
}

void Cell::SetNeighborsList(std::vector<unsigned long> & _v)
{
    m_NeighborsList = _v;
}

void Cell::SetVerticesList(std::vector<unsigned long> & _v)
{
    m_VerticesList = _v;
}

void Cell::SetJunctionsList(std::vector<unsigned long> & _v)
{
    m_JunctionsList = _v;
}

void Cell::SetPixelsList(std::vector< itk::Index<2> > & _v)
{
    m_PixelsList = _v;
}

void Cell::AddNeighbor(unsigned long & _v)
{
    std::vector<unsigned long>::iterator it;
    it = std::find(m_NeighborsList.begin(), m_NeighborsList.end(), _v);
    if (it == m_NeighborsList.end())
    {
        m_NeighborsList.push_back(_v);
    }
}

void Cell::AddJunction(unsigned long & _v)
{
    std::vector<unsigned long>::iterator it;
    it = std::find(m_JunctionsList.begin(), m_JunctionsList.end(), _v);
    if (it == m_JunctionsList.end())
    {
        m_JunctionsList.push_back(_v);
    }
}

void Cell::AddVertex(unsigned long & _v)
{
    std::vector<unsigned long>::iterator it;
    it = std::find(m_VerticesList.begin(), m_VerticesList.end(), _v);
    if (it == m_VerticesList.end())
    {
        m_VerticesList.push_back(_v);
    }
}

void Cell::AddPixel(itk::Index<2> & _v)
{
    std::vector< itk::Index<2> >::iterator it;
    it = std::find(m_PixelsList.begin(), m_PixelsList.end(), _v);
    if (it == m_PixelsList.end())
    {
        m_PixelsList.push_back(_v);
    }
}

float Cell::GetArea()
{
    return m_Area;
}

float Cell::GetPerimeter()
{
    return m_Perimeter;
}

float Cell::GetElongation()
{
    return m_Elongation;
}

float Cell::GetEccentricity()
{
    return m_Eccentricity;
}

float Cell::GetRoundness()
{
    return m_Roundness;
}

float Cell::GetAnisotropy()
{
    return m_Anisotropy;
}

float Cell::GetOrientation()
{
    return m_Orientation;
}

float Cell::GetMajorAxisLength()
{
    return m_MajorAxisLength;
}

float Cell::GetMinorAxisLength()
{
    return m_MinorAxisLength;
}

itk::FixedArray<float,2> Cell::GetCentroid()
{
    return m_Centroid;
}

float Cell::GetChordLength()
{
    return m_ChordLength;
}

float Cell::GetChordDisorder()
{
    return m_ChordDisorder;
}

double Cell::GetMaximumIntensity()
{
    return m_MaximumIntensity;
}

double Cell::GetMinimumIntensity()
{
    return m_MinimumIntensity;
}

double Cell::GetAverageIntensity()
{
    return m_AverageIntensity;
}

double Cell::GetVarianceIntensity()
{
    return m_VarianceIntensity;
}

float Cell::GetInertiaAnisotropy()
{
    return m_InertiaAnisotropy;
}
float Cell::GetInertiaOrientation()
{
    return m_InertiaOrientation;
}
float Cell::GetVertexOrientation()
{
    return m_VertexOrientation;
}
float Cell::GetVertexPolarity()
{
    return m_VertexPolarity;
}

void Cell::SetArea(float & _p)
{
    m_Area = _p;
}

void Cell::SetPerimeter(float & _p)
{
    m_Perimeter = _p;
}

void Cell::SetElongation(float & _p)
{
    m_Elongation = _p;
}

void Cell::SetEccentricity(float & _p)
{
    m_Eccentricity = _p;
}

void Cell::SetRoundness(float & _p)
{
    m_Roundness = _p;
}

void Cell::SetAnisotropy(float & _p)
{
    m_Anisotropy = _p;
}

void Cell::SetOrientation(float & _p)
{
    m_Orientation = _p;
}

void Cell::SetMajorAxisLength(float & _p)
{
    m_MajorAxisLength = _p;
}

void Cell::SetMinorAxisLength(float & _p)
{
    m_MinorAxisLength = _p;
}

void Cell::SetCentroid(float & _p1, float & _p2)
{
    m_Centroid[0] = _p1;
    m_Centroid[1] = _p2;
}

void Cell::SetMaximumIntensity(double & _p)
{
    m_MaximumIntensity = _p;
}

void Cell::SetMinimumIntensity(double & _p)
{
    m_MinimumIntensity = _p;
}

void Cell::SetAverageIntensity(double & _p)
{
    m_AverageIntensity = _p;
}

void Cell::SetVarianceIntensity(double & _p)
{
    m_VarianceIntensity = _p;
}

std::vector<float> Cell::GetInertiaTensor()
{
    return m_InertiaTensor;
}

std::vector<float> Cell::GetTextureTensor()
{
    return m_TextureTensor;
}

std::vector<float> Cell::GetVertexTensor()
{
    return m_VertexTensor;
}

double Cell::GetContourIntensity(unsigned int _p)
{
    if (_p < m_ContoursIntensityList.size())
        return m_ContoursIntensityList[_p];
    else
        return 0.0;
}

double Cell::GetContourIntensityDisorder(unsigned int _p)
{
    if (_p < m_ContoursIntensityDisordersList.size())
        return m_ContoursIntensityDisordersList[_p];
    else
        return 0.0;
}

double Cell::GetBackgroundIntensity()
{
    return m_BackgroundIntensity;
}

double Cell::GetBackgroundIntensity(unsigned int _p)
{
    if (_p < m_BackgroundIntensityList.size())
        return m_BackgroundIntensityList[_p];
    else
        return 0.0;
}


std::vector<double>  Cell::GetPolarityMode(unsigned int _p)
{
    return m_PolarityModesList[_p];
}

void Cell::ComputeIntensities(itk::Image<double, 2>::Pointer image, itk::Image<double, 2>::Pointer bg)
{
    m_BackgroundIntensity = 0;
    std::vector< itk::Index<2> >::const_iterator t;
    for (t = m_PixelsList.begin(); t != m_PixelsList.end(); ++t)
    {
        m_BackgroundIntensity += bg->GetPixel(*t);
    }
    m_BackgroundIntensity /= m_PixelsList.size();

    std::vector< double > intensityArray;
    m_ContourIntensity = 0;
    for (t = m_SmallDilPixelsList.begin(); t != m_SmallDilPixelsList.end(); ++t)
    {
        double i = image->GetPixel(*t) - m_BackgroundIntensity;
        if (i < 0)
        {
            i = 0;
        }
        intensityArray.push_back(i);
    }
    m_ContourIntensity = std::accumulate(intensityArray.begin(), intensityArray.end(), 0.0);
    m_ContourIntensity /= intensityArray.size();

    double meanInt = m_ContourIntensity;
    double sq_sum = std::inner_product(intensityArray.begin(), intensityArray.end(), intensityArray.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / intensityArray.size() - meanInt * meanInt) / meanInt;

    m_ContoursIntensityList.push_back(m_ContourIntensity);
    m_BackgroundIntensityList.push_back(m_BackgroundIntensity);
    m_ContoursIntensityDisordersList.push_back(stdev);
}

void Cell::ComputePolarityMode(itk::Image<double, 2>::Pointer image, itk::FixedArray<unsigned int, 2> & order, bool bg)
{
    std::vector<double> tmpPolarityMode(5);
    if (m_Category != CATEGORY::BORDERCELL)
    {
        int n_step = 18; // 20° arcs
        double thetaBinning[18] = { 0 };           // container to store intensity sum per segment
        double nbThetaBinning[18] = { 0 };         // container to store # of pixel used per segment
        double thetaList[18];                      // container to store average theta per segment (rad)
        double theta_step_rad = 2 * M_PI / n_step; // Theta steps in rad
        // Fill up the average theta per segment in rad
        for (int i = 0; i < n_step; i++)
        {
            thetaList[i] = i * theta_step_rad + theta_step_rad / 2;
        }
        // Compute average Intensity per theta segment
        std::vector< itk::Index<2> >::iterator it;
        for (it = m_SmallDilPixelsList.begin(); it != m_SmallDilPixelsList.end(); ++it)
        {
            double x = (*it)[order[0]] - m_Centroid[order[0]];
            double y = (*it)[order[1]] - m_Centroid[order[1]];
            double theta = std::atan2(y, x);     // return theta between [0 -> -PI]
            if (theta < 0)
            {
                theta += 2 * M_PI;               // return theta between [0 -> 2PI]
            }
            int idx = (int)std::floor(theta / theta_step_rad) % n_step;
            thetaBinning[idx] += image->GetPixel(*it);
            nbThetaBinning[idx] += 1;
        }
        // compute mode
        double A0(0), A1(0), A2(0), B1(0), B2(0);
        for (int i = 0; i < n_step; i++)
        {
            double avgIntensity = thetaBinning[i] / nbThetaBinning[i];
            A0 += avgIntensity;
            A1 += avgIntensity * std::cos(thetaList[i]);
            B1 += avgIntensity * std::sin(thetaList[i]);
            A2 += avgIntensity * std::cos(2 * thetaList[i]);
            B2 += avgIntensity * std::sin(2 * thetaList[i]);
        }
        // Apply constante and save
        double cst = theta_step_rad / M_PI;    // constante 
        tmpPolarityMode[0] = cst * A0 * 0.5;   // A0
        tmpPolarityMode[1] = cst * A1;         // A1
        tmpPolarityMode[2] = cst * B1;         // B1
        tmpPolarityMode[3] = cst * A2;         // A2
        tmpPolarityMode[4] = cst * B2;         // B2
        // substract background value if asked
        if (bg)
        {
            tmpPolarityMode[0] -= m_BackgroundIntensity;
        }
    }
    else
    {
        tmpPolarityMode[0] = NAN;
        tmpPolarityMode[1] = NAN;
        tmpPolarityMode[2] = NAN;
        tmpPolarityMode[3] = NAN;
        tmpPolarityMode[4] = NAN;
    }

    m_PolarityModesList.push_back(tmpPolarityMode);
    //std::cout << tmpPolarityMode[0] << " " << tmpPolarityMode[1] << " " << tmpPolarityMode[2] << " " << tmpPolarityMode[3] << " " << tmpPolarityMode[4] << std::endl;
}

std::vector< itk::Index<2> > Cell::GetSmallDilPixelsList()
{
    return m_SmallDilPixelsList;
}

void Cell::ComputeCategory(std::map<unsigned long, Cell> & _cList)
{
    std::vector<unsigned long>::const_iterator neighborsIt;
    if (m_Category == Cell::CATEGORY::CORECELL)
    {
        for (neighborsIt = m_NeighborsList.begin(); neighborsIt != m_NeighborsList.end(); ++neighborsIt)
        {
            if (_cList.find(*neighborsIt)->second.GetCategory() == 2)
            {
                m_Category = Cell::CATEGORY::FIRSTLAYER;
                break;
            }
        }
    }
}

void Cell::ComputeInertia(float _scale, itk::FixedArray<unsigned int, 2> & _order)
{
    if (m_Category != CATEGORY::BORDERCELL)
    {
        std::vector< itk::Index<2> >::const_iterator pixelsIt;
        float Ixx(0), Ixy(0), Iyy(0);
        for (pixelsIt = m_PixelsList.begin(); pixelsIt != m_PixelsList.end(); ++pixelsIt)
        {
            float x = (*pixelsIt)[_order[0]] - m_Centroid[_order[0]];
            float y = (*pixelsIt)[_order[1]] - m_Centroid[_order[1]];
            Ixx += x * x;
            Iyy += y * y;
            Ixy += x * y;
        }
        m_InertiaTensor[0] = Ixx / m_PixelsList.size();
        m_InertiaTensor[1] = Ixy / m_PixelsList.size();
        m_InertiaTensor[2] = Ixy / m_PixelsList.size();
        m_InertiaTensor[3] = Iyy / m_PixelsList.size();

        itk::FixedArray< float, 2 > eigenvalues;
        itk::Matrix< float, 2, 2 > eigenvectors;
        TensorData(m_InertiaTensor, eigenvalues, eigenvectors);

        m_InertiaAnisotropy = 1 - std::sqrt(eigenvalues[0] / eigenvalues[1]);  // small / big
        float angle = std::atan2(eigenvectors[1][1], eigenvectors[1][0]) * 180 / M_PI;
        if (angle > 90)  angle -= 180;
        if (angle < -90) angle += 180;
        m_InertiaOrientation = angle;

        m_InertiaTensor[0] *= std::pow(_scale, 2);
        m_InertiaTensor[1] *= std::pow(_scale, 2);
        m_InertiaTensor[2] *= std::pow(_scale, 2);
        m_InertiaTensor[3] *= std::pow(_scale, 2);
    }
    else
    {
        m_InertiaTensor[0] = NAN;
        m_InertiaTensor[1] = NAN;
        m_InertiaTensor[2] = NAN;
        m_InertiaTensor[3] = NAN;
    }
}

void Cell::ComputeTexture(std::map<unsigned long, Cell> & _cList, std::map<unsigned long, Junction> & _jList, float _scale, itk::FixedArray<unsigned int, 2> & _order)
{
    if (m_Category != CATEGORY::BORDERCELL)
    {
        float Mxx(0), Mxy(0), Myy(0);
        std::vector<unsigned long>::const_iterator neighborsIt;
        for (neighborsIt = m_NeighborsList.begin(); neighborsIt != m_NeighborsList.end(); ++neighborsIt)
        {
            float w = 1;
            unsigned long tmpJunctionLabel = CantorPairing(m_Label, *neighborsIt);
            float junctionFlag = _jList.find(tmpJunctionLabel)->second.GetChordLength();
            if (junctionFlag == 0)
            {
                w = 0.5;
            }
            itk::FixedArray<float, 2> o = _cList.find(*neighborsIt)->second.GetCentroid();
            float x = o[_order[0]] - m_Centroid[_order[0]];
            float y = o[_order[1]] - m_Centroid[_order[1]];
            Mxx += w * x * x;
            Myy += w * y * y;
            Mxy += w * x * y;
        }
        m_TextureTensor[0] = Mxx / 2 * std::pow(_scale, 2); // divide by two instead of divide per #neighbors
        m_TextureTensor[1] = Mxy / 2 * std::pow(_scale, 2); // see eq. A3 of "Tools" (Graner et al.)
        m_TextureTensor[2] = Mxy / 2 * std::pow(_scale, 2);
        m_TextureTensor[3] = Myy / 2 * std::pow(_scale, 2);
    }
    else
    {
        m_TextureTensor[0] = NAN;
        m_TextureTensor[1] = NAN;
        m_TextureTensor[2] = NAN;
        m_TextureTensor[3] = NAN;
    }
}

void Cell::ComputeVertices(std::map<unsigned long, Vertex> & _vList, itk::FixedArray<unsigned int, 2> & _order)
{
    if (m_Category != CATEGORY::BORDERCELL)
    {
        float Vxx(0), Vxy(0), Vyy(0);
        std::vector<unsigned long>::const_iterator vertexIt;
        for (vertexIt = m_VerticesList.begin(); vertexIt != m_VerticesList.end(); ++vertexIt)
        {
            itk::FixedArray < float, 2 > v = _vList.find(*vertexIt)->second.GetCoordinate();
            float x = v[_order[0]] - m_Centroid[_order[0]];
            float y = v[_order[1]] - m_Centroid[_order[1]];
            float norm = std::sqrt(std::pow(x, 2) + std::pow(y, 2));
            Vxx += (x * x) / (norm * norm);
            Vyy += (y * y) / (norm * norm);
            Vxy += (x * y) / (norm * norm);
        }
        m_VertexTensor[0] = Vxx;
        m_VertexTensor[1] = Vxy;
        m_VertexTensor[2] = Vxy;
        m_VertexTensor[3] = Vyy;

        itk::FixedArray< float, 2 > eigenvalues;
        itk::Matrix< float, 2, 2 > eigenvectors;
        TensorData(m_VertexTensor, eigenvalues, eigenvectors);

        m_VertexPolarity = 1 - (eigenvalues[0] / eigenvalues[1]);  // small / big
        float angle = std::atan2(eigenvectors[1][1], eigenvectors[1][0]) * 180 / M_PI;
        if (angle > 90)  angle -= 180;
        if (angle < -90) angle += 180;
        m_VertexOrientation = angle;
    }
    else
    {
        m_VertexTensor[0] = NAN;
        m_VertexTensor[1] = NAN;
        m_VertexTensor[2] = NAN;
        m_VertexTensor[3] = NAN;
    }
}

void Cell::ComputeChords(std::map<unsigned long, Junction> & _jList)
{
    float sumChord = 0;
    std::vector<float> chord(m_JunctionsList.size());
    std::vector<unsigned long>::const_iterator junctionsIt;
    for (junctionsIt = m_JunctionsList.begin(); junctionsIt != m_JunctionsList.end(); ++junctionsIt)
    {
        int i = junctionsIt - m_JunctionsList.begin();
        float currentChord = _jList.find(*junctionsIt)->second.GetChordLength();
        chord[i] = currentChord;
        sumChord += currentChord;
    }
    float meanChord = sumChord / chord.size();
    float sq_sum = std::inner_product(chord.begin(), chord.end(), chord.begin(), 0.0);
    float stdev = std::sqrt(sq_sum / chord.size() - meanChord * meanChord) / meanChord;

    m_ChordLength = sumChord;
    m_ChordDisorder = stdev;
}

void Cell::ComputeJunctionsPixelsList(std::map<unsigned long, Junction> & _jList)
{
    std::vector<unsigned long>::const_iterator junctionsIt;
    for (junctionsIt = m_JunctionsList.begin(); junctionsIt != m_JunctionsList.end(); ++junctionsIt)
    {
        std::vector< itk::Index<2> > normal = _jList.find(*junctionsIt)->second.GetPixelsList();
        std::vector< itk::Index<2> > dilate = _jList.find(*junctionsIt)->second.GetSmallDilPixelsList();
        m_SmallDilPixelsList.insert(m_SmallDilPixelsList.end(), dilate.begin(), dilate.end());
        m_JunctionsPixelsList.insert(m_JunctionsPixelsList.end(), normal.begin(), normal.end());
    }
    m_SmallDilPixelsList.erase(std::unique(m_SmallDilPixelsList.begin(), m_SmallDilPixelsList.end()), m_SmallDilPixelsList.end());
    m_JunctionsPixelsList.erase(std::unique(m_JunctionsPixelsList.begin(), m_JunctionsPixelsList.end()), m_JunctionsPixelsList.end());
}



std::vector<unsigned long> Cell::GetIndicesPixelsList()//(int _w, int _h)
{
    std::vector<unsigned long> res;
    std::vector< itk::Index<2> >::const_iterator it;
    for (it = m_PixelsList.begin(); it != m_PixelsList.end(); ++it)
    {
        res.push_back(sub2ind(WIDTH, HEIGHT, *it));
    }
    return res;
}

std::vector<unsigned long> Cell::GetJunctionsIndicesPixelsList()//(int _w, int _h)
{
    std::vector<unsigned long> res;
    std::vector< itk::Index<2> >::const_iterator it;
    for (it = m_JunctionsPixelsList.begin(); it != m_JunctionsPixelsList.end(); ++it)
    {
        res.push_back(sub2ind(WIDTH, HEIGHT, *it));
    }
    return res;
}

std::vector<unsigned long> Cell::GetDilJunctionsIndicesPixelsList()//(int _w, int _h)
{
    std::vector<unsigned long> res;
    std::vector< itk::Index<2> >::const_iterator it;
    for (it = m_SmallDilPixelsList.begin(); it != m_SmallDilPixelsList.end(); ++it)
    {
        res.push_back(sub2ind(WIDTH, HEIGHT, *it));
    }
    return res;
}



#endif //__CELL_H