/**
* \class Vertex
*
*
* \brief Class regrouping all information related to a vertex in a tissu
*
* This class gathered all information related to a vertex, its position
* in the image, but also its relationship with other component of a tissu,
* such as junctions and cells.
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

#ifndef __VERTEX_H
#define __VERTEX_H

#include "itkImage.h"
#include "Utility.h"

class Vertex 
{

public:

    typedef unsigned long    LabelType;
    typedef itk::Index < 2 > IndexType;
    typedef itk::FixedArray < float, 2 > PointType;

    Vertex();
    Vertex(LabelType &);
    ~Vertex();

    LabelType GetLabel();
    void SetLabel(LabelType &);

    bool isNeighbor(LabelType &);
    void AddCell(LabelType &);
    bool operator == (const Vertex &) const;

    std::vector < LabelType > GetCellList();
    PointType GetCoordinate();
    LabelType GetLinearIndex();

    void SetCoordinate(PointType &);
    void SetCoordinate(float, float);
    void SetLinearIndex(LabelType &);
    void SetCellList(std::vector < LabelType > &);

private:

    LabelType m_Label;
    LabelType m_LinearIndex;
    PointType m_Coordinate;
    std::vector < LabelType > m_CellList;

}; // end of class Vertex



Vertex::Vertex()
{}

Vertex::Vertex(unsigned long & _p) : m_Label(_p)
{}

Vertex::~Vertex()
{}

unsigned long Vertex::GetLabel()
{
    return m_Label;
}

void Vertex::SetLabel(unsigned long & _p)
{
    m_Label = _p;
}

bool Vertex::operator==(const Vertex & _p) const
{
    return this->m_Label == _p.m_Label;
}

bool Vertex::isNeighbor(unsigned long & _p)
{
    bool res = false;
    std::vector<unsigned long>::iterator it;
    it = std::find(m_CellList.begin(), m_CellList.end(), _p);
    if (it == m_CellList.end())
    {
        res = true;
    }
    return res;
}

void Vertex::AddCell(unsigned long & _v)
{
    std::vector<unsigned long>::iterator it;
    it = std::find(m_CellList.begin(), m_CellList.end(), _v);
    if (it == m_CellList.end())
    {
        m_CellList.push_back(_v);
    }
}

std::vector<unsigned long> Vertex::GetCellList()
{
    return m_CellList;
}

void Vertex::SetCellList(std::vector<unsigned long> & _p)
{
    m_CellList = _p;
}

itk::FixedArray < float, 2 > Vertex::GetCoordinate()
{
    return m_Coordinate;
}

unsigned long Vertex::GetLinearIndex()
{
    return m_LinearIndex;
}

void Vertex::SetCoordinate(itk::FixedArray < float, 2 > & _p)
{
    m_Coordinate = _p;
}

void Vertex::SetCoordinate(float _x, float _y)
{
    m_Coordinate[0] = _x;
    m_Coordinate[1] = _y;
}

void Vertex::SetLinearIndex(unsigned long & _p)
{
    m_LinearIndex = _p;
}


#endif // __VERTEX_H