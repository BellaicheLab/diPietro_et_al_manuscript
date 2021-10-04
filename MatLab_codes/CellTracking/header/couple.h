#ifndef __couple_h__
#define __couple_h__

class Couple
{
    unsigned long m_pixelvalue;
    int m_number;
    int m_centroid_is_in_the_cell_n;
    int m_centroid_is_in_the_own_cell;
public:
    Couple();
    void UnDePlus ();
    unsigned long GetPixelValue();
    int GetNumber();
    void SetPixelValue(unsigned long);
    void CentroidIsInTheCellN();
    void CentroidIsInTheOwnCell();
    void SetNumber(int);
    int GetCentroidIsInTheCellN();
    int GetCentroidIsInTheOwnCell();
    void Delete();
};

#endif // __couple_h__
