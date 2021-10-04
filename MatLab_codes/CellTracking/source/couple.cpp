
#include <cstdlib>

#include "couple.h"

using namespace std;

Couple::Couple()
{
    m_number = 1;
    m_pixelvalue = 0;
    m_centroid_is_in_the_cell_n = 0;
    m_centroid_is_in_the_own_cell = 0;
}

int Couple::GetNumber()
{
    return m_number;
}

unsigned long Couple::GetPixelValue()
{
    return m_pixelvalue;
}

void Couple::UnDePlus()
{
    m_number = m_number + 1;
}

void Couple::SetNumber(int a)
{
    m_number = a;
}

void Couple::SetPixelValue(unsigned long a)
{
    m_pixelvalue = a;
}

void Couple::CentroidIsInTheCellN()
{
    m_centroid_is_in_the_cell_n = 1;
}

int Couple::GetCentroidIsInTheCellN()
{
    return m_centroid_is_in_the_cell_n;
}

void Couple::CentroidIsInTheOwnCell()
{
    m_centroid_is_in_the_own_cell = 1;
}

int Couple::GetCentroidIsInTheOwnCell()
{
    return m_centroid_is_in_the_own_cell;
}

void Couple::Delete()
{
    m_number = 0;
    m_pixelvalue = 0;
    m_centroid_is_in_the_cell_n = 0;
    m_centroid_is_in_the_own_cell = 0;
}
