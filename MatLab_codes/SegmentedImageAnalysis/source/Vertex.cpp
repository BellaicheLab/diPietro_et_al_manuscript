
#include "Vertex.h"


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

itk::Index<2> Vertex::GetCoordinate()
{
    return m_Coordinate;
}

unsigned long Vertex::GetLinearIndex()
{
    return m_LinearIndex;
}

void Vertex::SetCoordinate(itk::Index<2> & _p)
{
    m_Coordinate = _p;
}

void Vertex::SetLinearIndex(unsigned long & _p)
{
    m_LinearIndex = _p;
}


