#include "trackingfunction.h"

/* Add a neighbour in the neighbour map
* @param neighbourMap a pointer to the neighbour map
* @param currentCell the cell RN
* @param neighbour1 the first neighbour RN
* @param neighbour2 the second neighbour RN (optional)
* @param neighbour3 the third neighbour RN (optional)
*/
void
AddNeighboursInMap(std::map<unsigned long, std::vector<unsigned long> > *neighbourMap, unsigned long currentCell, unsigned long neighbour1, unsigned long neighbour2 = 0, unsigned long neighbour3 = 0)
{
    // we search the neighbour map for the cell RN #i
    std::map<unsigned long, std::vector<unsigned long> >::iterator it = neighbourMap->find(currentCell);
    if (it != neighbourMap->end()) // we found the cell
    {
        std::vector<unsigned long>::iterator itv = find(it->second.begin(), it->second.end(), neighbour1);
        if (itv == it->second.end())
        {
            it->second.push_back(neighbour1); // add neighbour1 to the list if not already in
        }
        if (neighbour2 != 0)
        {
            itv = find(it->second.begin(), it->second.end(), neighbour2);
            if (itv == it->second.end())
            {
                it->second.push_back(neighbour2); // add neighbour2 to the list if not already in
            }
            if (neighbour3 != 0)
            {
                itv = find(it->second.begin(), it->second.end(), neighbour3);
                if (itv == it->second.end())
                {
                    it->second.push_back(neighbour3); // add neighbour3 to the list if not already in
                }
            }
        }
    }
    else // we did not found the cell
    {
        // create a new entry in the map and add the neighbours 
        std::vector<unsigned long> neighbours;
        neighbours.push_back(neighbour1);
        if (neighbour2 != 0)
        {
            neighbours.push_back(neighbour2);
        }
        if (neighbour3 != 0)
        {
            neighbours.push_back(neighbour3);
        }
        neighbourMap->insert(std::make_pair(currentCell, neighbours));
    }
}

/* Fill the neighbour map and print the skeleton pixels info file
* @param labeledITKimage a pointer to the labeled image
* @param neighbourMap a pointer to the map to fill
* @param outputfilename the name of the output file
* @param bordercells a reference to the RN list of the border cells
*/
void
FillNeighboursList(const ITKImageType::Pointer &labeledITKimage, std::map<unsigned long, std::vector<unsigned long> > *neighbourMap, std::string outputfilename, std::vector<unsigned long> &bordercells)
{
    // print the output in a file (2012-05-02)
    std::ofstream skelpixfile;
    if (!outputfilename.empty())
    {
        skelpixfile.open(outputfilename.c_str());
    }

    // in : iterator on the labeled image
    ConstIteratorType iteImage(labeledITKimage, labeledITKimage->GetLargestPossibleRegion());
    ConstNeighborhoodIteratorType::RadiusType radius;
    radius.Fill(1);
    // it : neighborhood iterator to iterate on the neighbor pixels of the in iterator
    ConstNeighborhoodIteratorType iteNeighborhood(radius, labeledITKimage, labeledITKimage->GetLargestPossibleRegion());
    // Definition of the neighbor pixels on which the iteration will be done
    ConstNeighborhoodIteratorType::OffsetType offset1 = { { -1, -1 } };
    ConstNeighborhoodIteratorType::OffsetType offset2 = { { -1, 0 } };
    ConstNeighborhoodIteratorType::OffsetType offset3 = { { -1, 1 } };
    ConstNeighborhoodIteratorType::OffsetType offset4 = { { 0, -1 } };
    ConstNeighborhoodIteratorType::OffsetType offset5 = { { 0, 1 } };
    ConstNeighborhoodIteratorType::OffsetType offset6 = { { 1, -1 } };
    ConstNeighborhoodIteratorType::OffsetType offset7 = { { 1, 0 } };
    ConstNeighborhoodIteratorType::OffsetType offset8 = { { 1, 1 } };

    // Iteration on the skeleton of the labeled image
    unsigned long a(0), b(0), c(0), d(0);
    unsigned long cptPixel = 0;
    for (iteImage.GoToBegin(), iteNeighborhood.GoToBegin(); !iteImage.IsAtEnd(); ++iteImage, ++iteNeighborhood, cptPixel++)
    {
        if (iteImage.Get() == 0) // we ensures that the iteration is made on the skeleton pixels (skel pixel = 0)
        {
            if (iteNeighborhood.GetPixel(offset1) != 0) // check offset1
            {
                a = iteNeighborhood.GetPixel(offset1);
            }
            if (iteNeighborhood.GetPixel(offset2) != 0) // check offset2
            {
                if (a == 0)
                {
                    a = iteNeighborhood.GetPixel(offset2);
                }
                else if (iteNeighborhood.GetPixel(offset2) != a)
                {
                    b = iteNeighborhood.GetPixel(offset2);
                }
            }
            if (iteNeighborhood.GetPixel(offset3) != 0) // check offset3
            {
                if (a == 0)
                {
                    a = iteNeighborhood.GetPixel(offset3);
                }
                else if (iteNeighborhood.GetPixel(offset3) != a)
                {
                    if (b == 0)
                    {
                        b = iteNeighborhood.GetPixel(offset3);
                    }
                    else if (iteNeighborhood.GetPixel(offset3) != b)
                    {
                        c = iteNeighborhood.GetPixel(offset3);
                    }
                }
            }
            if (iteNeighborhood.GetPixel(offset4) != 0) // check offset4
            {
                if (a == 0)
                {
                    a = iteNeighborhood.GetPixel(offset4);
                }
                else if (iteNeighborhood.GetPixel(offset4) != a)
                {
                    if (b == 0)
                    {
                        b = iteNeighborhood.GetPixel(offset4);
                    }
                    else if (iteNeighborhood.GetPixel(offset4) != b)
                    {
                        if (c == 0)
                        {
                            c = iteNeighborhood.GetPixel(offset4);
                        }
                        else if (iteNeighborhood.GetPixel(offset4) != c)
                        {
                            d = iteNeighborhood.GetPixel(offset4);
                        }
                    }
                }
            }
            if (iteNeighborhood.GetPixel(offset5) != 0 && d == 0) //check offset5
            {
                if (a == 0)
                {
                    a = iteNeighborhood.GetPixel(offset5);
                }
                else if (iteNeighborhood.GetPixel(offset5) != a)
                {
                    if (b == 0)
                    {
                        b = iteNeighborhood.GetPixel(offset5);
                    }
                    else if (iteNeighborhood.GetPixel(offset5) != b)
                    {
                        if (c == 0)
                        {
                            c = iteNeighborhood.GetPixel(offset5);
                        }
                        else if (iteNeighborhood.GetPixel(offset5) != c)
                        {
                            d = iteNeighborhood.GetPixel(offset5);
                        }
                    }
                }
            }
            if (iteNeighborhood.GetPixel(offset6) != 0 && d == 0) //check offset6
            {
                if (a == 0)
                {
                    a = iteNeighborhood.GetPixel(offset6);
                }
                else if (iteNeighborhood.GetPixel(offset6) != a)
                {
                    if (b == 0)
                    {
                        b = iteNeighborhood.GetPixel(offset6);
                    }
                    else if (iteNeighborhood.GetPixel(offset6) != b)
                    {
                        if (c == 0)
                        {
                            c = iteNeighborhood.GetPixel(offset6);
                        }
                        else if (iteNeighborhood.GetPixel(offset6) != c)
                        {
                            d = iteNeighborhood.GetPixel(offset6);
                        }
                    }
                }
            }
            if (iteNeighborhood.GetPixel(offset7) != 0 && d == 0) //check offset7
            {
                if (a == 0)
                {
                    a = iteNeighborhood.GetPixel(offset7);
                }
                else if (iteNeighborhood.GetPixel(offset7) != a)
                {
                    if (b == 0)
                    {
                        b = iteNeighborhood.GetPixel(offset7);
                    }
                    else if (iteNeighborhood.GetPixel(offset7) != b)
                    {
                        if (c == 0)
                        {
                            c = iteNeighborhood.GetPixel(offset7);
                        }
                        else if (iteNeighborhood.GetPixel(offset7) != c)
                        {
                            d = iteNeighborhood.GetPixel(offset7);
                        }
                    }
                }
            }
            if (iteNeighborhood.GetPixel(offset8) != 0 && d == 0) //check offset8
            {
                if (a == 0)
                {
                    a = iteNeighborhood.GetPixel(offset8);
                }
                else if (iteNeighborhood.GetPixel(offset8) != a)
                {
                    if (b == 0)
                    {
                        b = iteNeighborhood.GetPixel(offset8);
                    }
                    else if (iteNeighborhood.GetPixel(offset8) != b)
                    {
                        if (c == 0)
                        {
                            c = iteNeighborhood.GetPixel(offset8);
                        }
                        else if (iteNeighborhood.GetPixel(offset8) != c)
                        {
                            d = iteNeighborhood.GetPixel(offset8);
                        }
                    }
                }
            }
            if (b != 0) // to prevent 1-neighbour cases (a.k.a. bad skeletonization case)
            {
                if (c == 0) // dual case
                {
                    AddNeighboursInMap(neighbourMap, a, b); // a is a neighbour of b
                    AddNeighboursInMap(neighbourMap, b, a); // b is a neighbour of a
                }
                else if (d == 0) // triple case
                {
                    AddNeighboursInMap(neighbourMap, a, b, c); // a is a neighbour of {b,c}
                    AddNeighboursInMap(neighbourMap, b, a, c); // b is a neighbour of {a,c}
                    AddNeighboursInMap(neighbourMap, c, a, b); // c is a neighbour of {a,b}
                }
                else // quad case
                {
                    AddNeighboursInMap(neighbourMap, a, b, c, d); // a is a neighbour of {b,c,d}
                    AddNeighboursInMap(neighbourMap, b, a, c, d); // b is a neighbour of {a,c,d}
                    AddNeighboursInMap(neighbourMap, c, a, b, d); // c is a neighbour of {a,b,d}
                    AddNeighboursInMap(neighbourMap, d, a, b, c); // d is a neighbour of {a,b,c}
                }
            }
            if (!outputfilename.empty()) // write skeleton pixel into file
            {
                skelpixfile << cptPixel << " " << a << " " << b << " " << c << " " << d << std::endl;
            }
            a = b = c = d = 0; // reset
        }
        else // on a cell
        {
            // if on the image border & not already seen
            if ((iteImage.GetIndex()[0] == 0 || iteImage.GetIndex()[1] == 0 ||
                iteImage.GetIndex()[0] == labeledITKimage->GetLargestPossibleRegion().GetSize()[0] - 1 ||
                iteImage.GetIndex()[1] == labeledITKimage->GetLargestPossibleRegion().GetSize()[1] - 1) &&
                find(bordercells.begin(), bordercells.end(), iteImage.Get()) == bordercells.end())
            {
                bordercells.push_back(iteImage.Get());
            }
        }
    }
    // close skeleton pixel file
    if (!outputfilename.empty())
    {
        skelpixfile.close();
    }
}

/* Print a neighbour map
* @param filename the output file name
* @param neighbourMap the map to print
*/
void
PrintNeighboursMap(std::string filename, const std::map<unsigned long, std::vector<unsigned long> > &neighbourMap)
{
    std::ofstream neighbourfile(filename.c_str());
    std::map<unsigned long, std::vector<unsigned long> >::const_iterator iteNeighbour;
    std::vector<unsigned long>::const_iterator                      iteInnerCell;
    for (iteNeighbour = neighbourMap.cbegin(); iteNeighbour != neighbourMap.cend(); iteNeighbour++)
    {
        neighbourfile << iteNeighbour->first;
        for (iteInnerCell = iteNeighbour->second.cbegin(); iteInnerCell != iteNeighbour->second.cend(); iteInnerCell++)
        {
            neighbourfile << " " << *iteInnerCell;
        }
        neighbourfile << std::endl;
    }
    neighbourfile.close();
}

/* Read border cells file contents and store it in a vector.
* @param filename the file name
* @param bordercells the destination vector
*/
void
ReadBorderCellsFile(std::string filename, std::vector<unsigned long> &bordercells)
{
    std::ifstream borderfile(filename.c_str());
    if (borderfile.is_open())
    {
        // read file content
        std::string line;
        while (borderfile.good())
        {
            getline(borderfile, line); // read the line
            if (!line.empty())
            {
                // line in the std::stringstream
                std::stringstream ss(line);
                unsigned long tmp;
                ss >> tmp;
                bordercells.push_back(tmp);
            }
        }
        borderfile.close(); // close file before leaving
    }
    else
    {
        std::cerr << "ERROR in 'ReadBorderCellsFile': Unable to open " << filename << std::endl;
    }
}

/* Print border cells in file.
* @param filename the file name
* @param bordercells the destination vector
*/
void
PrintBordercells(std::string filename, const std::vector<unsigned long> &bordercells)
{
    std::ofstream border(filename.c_str());
    std::vector<unsigned long>::const_iterator iteBorderCells;
    for (iteBorderCells = bordercells.begin(); iteBorderCells != bordercells.end(); iteBorderCells++)
    {
        border << *iteBorderCells << std::endl;
    }
    border.close();
}

/* Find Minimum in an array
* @param tab the array
* @param nbElement the number of elements in the array
* @return the index of the Minimum in the array
*/
unsigned int
Minimum(const double *tab, unsigned int nbElement)
{
    unsigned int minIndex = 0;
    for (unsigned int i = 1; i < nbElement; i++)
    {
        if (tab[i] < tab[minIndex])
        {
            minIndex = i;
        }
    }
    return minIndex;
}

/* Find the index of the minimum in an array
* @param tab the array
* @return the index of the Minimum in the array
*/
unsigned int
MinimumIndex(const std::vector<double> tab)
{
    unsigned int minIndex = 0;
    std::vector<double>::const_iterator iteTab;
    for (iteTab = tab.begin(); iteTab != tab.end(); ++iteTab)
    {
        if (*iteTab < tab.at(minIndex))
        {
            minIndex = iteTab - tab.begin();
        }
    }
    return minIndex;
}

/* Sort couple by number
* @param couple1 the first couple
* @param couple2 the second couple
* @return true if couple1 < couple2
*/
bool
CompareByNumber(Couple couple1, Couple couple2)
{
    return (couple1.GetNumber() < couple2.GetNumber());
}

/* Shift coordonate point (x,y) according to PIV (u,v)
* @param cIndex the coordinate index to be shifted
* @param itePIV1 the iterator on the PIV_U
* @param itePIV2 the iterator on the PIV_V
* @param imageSize the image size
* @param bias additionnal bias
* @return new index
*/
ITKImageType::IndexType
ShiftPointByPIV(ITKImageType::IndexType &cIndex, IteratorType &itePIV1, IteratorType &itePIV2, ITKImageType::SizeType &imageSize, bool old)
{
    long movedX, movedY;
    ITKImageType::IndexType mIndex;
    mIndex[0] = 0;
    mIndex[1] = 0;

    // get PIV value
    itePIV1.SetIndex(cIndex);        // PIV value (U)
    itePIV2.SetIndex(cIndex);        // PIV value (V)
    int pivU = itePIV1.Get() - 128;  // -128 because PIV are saved in [0 255]
    int pivV = itePIV2.Get() - 128;  // -128 because PIV are saved in [0 255]

    if (old) // we move from the previous frame to the current frame
    {
        movedX = (int)cIndex[0] + pivU;
        movedY = (int)cIndex[1] + pivV;
    }
    else // we move from the current frame to the previous frame
    {
        movedX = (int)cIndex[0] - pivU;
        movedY = (int)cIndex[1] - pivV;
    }

    // out of bound case X
    if (movedX < 0)
    {
        movedX = 0;
    }
    else if (movedX > imageSize[0])
    {
        movedX = imageSize[0] - 1;
    }
    // set x after move
    mIndex[0] = movedX;

    // out of bound case Y
    if (movedY < 0)
    {
        movedY = 0;
    }
    else if (movedY >= imageSize[1])
    {
        movedY = imageSize[1] - 1;
    }
    // set y after move
    mIndex[1] = movedY;

    return mIndex;
}

/* Calculate cell overlap between  previous and current frames
* @param overlap the overlap map to be calculate
* @param overlapFiller the overlap data structure
* @param iteImage1 the iterator on the first frame
* @param iteImage2 the iterator on the second frame
* @param itePIV1 the iterator on the PIV_U
* @param itePIV2 the iterator on the PIV_V
* @param imageSize the image size
*/
void
OverlapCell2CellComparison(std::map<int, std::vector<Couple> > &overlap, Couple &overlapFiller, ITKImageType *image1, ITKImageType *image2, ITKImageType *PIV1, ITKImageType *PIV2, ITKImageType::SizeType &imageSize)
{
    // image iterators
    IteratorType iteImage1(image1, image1->GetLargestPossibleRegion());
    IteratorType iteImage2(image2, image2->GetLargestPossibleRegion());
    // piv iterators
    IteratorType itePIV1(PIV1, PIV1->GetLargestPossibleRegion());
    IteratorType itePIV2(PIV2, PIV2->GetLargestPossibleRegion());

    for (iteImage1.GoToBegin(); !iteImage1.IsAtEnd(); ++iteImage1)
    {
        if (iteImage1.Get() != 0)
        {
            // shift the pixel coordinates
            ITKImageType::IndexType cIndex = iteImage1.GetIndex();
            ITKImageType::IndexType mIndex;
            mIndex = ShiftPointByPIV(cIndex, itePIV1, itePIV2, imageSize, true);
            iteImage2.SetIndex(mIndex);

            // calculate overlap
            if (iteImage2.Get() != 0)
            {
                bool pixelAlreadyRegistred = false;
                std::vector<Couple>::iterator iteOverlap;
                for (iteOverlap = overlap[iteImage1.Get()].begin(); iteOverlap != overlap[iteImage1.Get()].end(); ++iteOverlap)
                {
                    if (iteImage2.Get() == iteOverlap->GetPixelValue())
                    {
                        iteOverlap->UnDePlus();
                        pixelAlreadyRegistred = true;
                        break;
                    }
                }
                if (!pixelAlreadyRegistred)
                {
                    overlapFiller.Delete();
                    overlapFiller.SetPixelValue(iteImage2.Get());
                    overlap[iteImage1.Get()].push_back(overlapFiller);
                }
            }
        }
    } // end loop for image1 pixel
}

/* Detected centroids-overlap couple between previous and current frames
* @param centroidsArray the centroids table to be calculate
* @param numberOfCells2 the number of cell in the second frames
* @param bestOverlap the sorted map of overlap ration
* @param geometryFilter1 the geometry map of the frame 1
* @param geometryFilter2 the geometry map of the frame 2
* @param iteImage1 the iterator on the first frame
* @param iteImage2 the iterator on the second frame
* @param itePIV1 the iterator on the PIV_U
* @param itePIV2 the iterator on the PIV_V
* @param imageSize the image size
*/
void
CentroidToCellOverlap(std::pair<double, double> *centroidsArray, unsigned long numberOfCells2, std::map<int, std::vector<Couple> > &bestOverlap, GeometryFilterType *geometryFilter1, GeometryFilterType *geometryFilter2, IteratorType &iteImage1, IteratorType &iteImage2, IteratorType &itePIV1, IteratorType &itePIV2, ITKImageType::SizeType &imageSize)
{
    // fill up the centroids array
    for (unsigned long currentCell = 0; currentCell < numberOfCells2; currentCell++)
    {
        double cx = geometryFilter2->GetCentroid(currentCell + 1)[0];  // centroid x
        double cy = geometryFilter2->GetCentroid(currentCell + 1)[1];  // centroid y
        if (cx < 0 || cy < 0)  // any negative value? (should not happen but ...)
        {
            std::cerr << "WARNING 'CentroidToCellOverlap' : negative centroid for area#" << currentCell + 1 << " (" << cx << ";" << cy << ")" << std::endl;
            cx = (cx < 0) ? -cx : cx;
            cy = (cy < 0) ? -cy : cy;
        }
        centroidsArray[currentCell] = std::make_pair(cx, cy);  // add in the array
    }

    // shift all centroids from image2 (previous) to image1 (current) using PIV
    for (unsigned long currentCell = 0; currentCell < numberOfCells2; currentCell++)
    {
        // shift the current centroid
        ITKImageType::IndexType mIndex;
        ITKImageType::IndexType cIndex;
        cIndex[0] = roundToInt(centroidsArray[currentCell].first);
        cIndex[1] = roundToInt(centroidsArray[currentCell].second);
        mIndex = ShiftPointByPIV(cIndex, itePIV1, itePIV2, imageSize, false);
        // set position on centroids in images
        iteImage1.SetIndex(mIndex);
        iteImage2.SetIndex(cIndex);
        // get area number
        unsigned long pixelValue1 = iteImage1.Get();
        unsigned long pixelValue2 = iteImage2.Get();

        // is in the cell ?
        std::vector<Couple>::iterator iteCouple;
        for (iteCouple = bestOverlap[pixelValue1].begin(); iteCouple != bestOverlap[pixelValue1].end(); iteCouple++)
        {
            if (iteCouple->GetPixelValue() == pixelValue2)
            {
                iteCouple->CentroidIsInTheCellN();
                break;
            }
        }
    }  // end for loop moving centroids from image2 to image1 using PIV
}

/* Determine if the division is valide or not
* @param firstCell the first hypothetique daugter
* @param secondCell the second hypothetique daugter
* @param currentCell the mother cell
* @param neighbourMap the neightbour map
* @param anisotropy the anistropy values
* @param geometryFilter1 the geometry map of frame 1
* @param geometryFilter2 the geometry map of frame 2
* @param borderCells1 the border cell map of frame 1
* @param borderCells2 the border cell map of frame 2
* @param daughterDaughterRatio2 the daughter size comparison value
* @param motherDaughterRatio2 the mother-daughter size comparison value
* @param anisotropyThreshold the anisotropy comparison value
* @param debugMode debug flag
* @param dlog debug output file
* @return true if all test are successfull
*/
bool
DivisionVerificationFunction(unsigned long firstCell, unsigned long secondCell, unsigned long currentCell, std::map<int, std::map<int, std::vector<CorrespondanceCouple> > > &correspondance, std::map<unsigned long, std::vector<unsigned long> > &neighbourMap, std::map<unsigned long, double> &anisotropy, GeometryFilterType *geometryFilter1, GeometryFilterType *geometryFilter2, std::vector<unsigned long> &borderCells1, std::vector<unsigned long> &borderCells2, double daughterDaughterRatio2, double motherDaughterRatio2, double anisotropyThreshold, unsigned int maximumDivisionAllowed, unsigned int previousFrame, bool debugMode, std::ofstream &dlog)
{
    bool division = true;

    // division verification filter 0 : is number of division max reached
    std::string motherTag = correspondance[previousFrame][currentCell].front().GetTag();
    unsigned int motherNbDivision = motherTag.length() / 2;
    if (motherNbDivision >= maximumDivisionAllowed)
    {
        division = false;
        if (debugMode)
        {
            dlog << "\tdivision verification filter 0 - maximum division filter: " << division << std::endl;
        }
    }
    // division verification filter 1 : daughter is a neighbours
    if (division)
    {
        std::map<unsigned long, std::vector<unsigned long> >::iterator itNeighborMap = neighbourMap.find(firstCell);
        if (itNeighborMap != neighbourMap.end())
        {
            // not found
            if (find(itNeighborMap->second.begin(), itNeighborMap->second.end(), secondCell) == itNeighborMap->second.end())
            {
                division = false;
                if (debugMode)
                {
                    dlog << "\tdivision verification filter 1 - neighbour filter: " << division << std::endl;
                }
            }
        }
    }
    // division verification filter 2 : daugther - daughter ratio
    if (division)
    {
        if (geometryFilter2->GetVolume(firstCell) < geometryFilter2->GetVolume(secondCell))
        {
            if (geometryFilter2->GetVolume(firstCell) / (double)geometryFilter2->GetVolume(secondCell) < daughterDaughterRatio2)
            {
                division = false;
                if (debugMode)
                {
                    dlog << "\tdivision verification filter 2 - D&D ratio filter: " << division << std::endl;
                }
            }
        }
        else
        {
            if (geometryFilter2->GetVolume(secondCell) / (double)geometryFilter2->GetVolume(firstCell) < daughterDaughterRatio2)
            {
                division = false;
                if (debugMode)
                {
                    dlog << "\tdivision verification filter 2 - D&D ratio filter: " << division << std::endl;
                }
            }
        }
    }
    // division verification filter 3 : mother - daughter ratio
    if (division && (geometryFilter2->GetVolume(firstCell) / (double)geometryFilter1->GetVolume(currentCell) > motherDaughterRatio2
        || geometryFilter2->GetVolume(secondCell) / (double)geometryFilter1->GetVolume(currentCell) > motherDaughterRatio2))
    {
        division = false;
        if (debugMode)
        {
            dlog << "\tdivision verification filter 3 - M&D ratio filter: " << division << std::endl;
        }
    }
    // division verification filter 4 : anisotropy (3.6)
    if (division && (anisotropy[firstCell] > anisotropyThreshold || anisotropy[secondCell] > anisotropyThreshold))
    {
        division = false;
        if (debugMode)
        {
            dlog << "\tdivision verification filter 4 - anisotropy filter: " << division << std::endl;
        }
    }
    // division verification filter 5a :  border cell mother
    if (division && find(borderCells1.begin(), borderCells1.end(), currentCell) != borderCells1.end())
    {
        division = false;
        if (debugMode)
        {
            dlog << "\tdivision verification filter 5a - bordercell 1 filter: " << division << std::endl;
        }
    }
    // division verification filter 5b :  border cell daughter
    std::vector<unsigned long>::iterator iteBordeCell;
    for (iteBordeCell = borderCells2.begin(); division && iteBordeCell != borderCells2.end(); iteBordeCell++)
    {
        if (*iteBordeCell == firstCell || *iteBordeCell == secondCell)
        {
            division = false;
            if (debugMode)
            {
                dlog << "\tdivision verification filter 5b - bordercell 2 filter: " << division << std::endl;
            }
        }
    }
    if (division && debugMode)
    {
        dlog << "\tdivision verification filter all valide" << std::endl;
    }
    return division;
}

/* True if a region (RN) is listed as newly coalesced elements
* NB: a region (RN) is newly coalesced if it containes at least one newly coalesced tag (AN)
* @param newlyCoalescedElements the map of newly coalesced region and tag
* @param testCell the RN to search for
* @return true if found
*/
bool IsCoalesced(std::map<unsigned long, std::vector<std::string> > &newlyCoalescedElements, unsigned long testCell)
{
    bool found = false;
    std::map<unsigned long, std::vector<std::string> >::iterator iteNewCoalesced;
    for (iteNewCoalesced = newlyCoalescedElements.begin(); iteNewCoalesced != newlyCoalescedElements.end(); ++iteNewCoalesced)
    {
        if (iteNewCoalesced->first == testCell)
        {
            found = true; // we found a coalesced cell has neighbor
            break;
        }
    }
    return found;
}

/* True if a region (RN) is listed as just divided cell
* @param justDividedCellsRN2 the list of newly divided region (RN)
* @param testCell the RN to search for
* @return true if found
*/
bool IsNewlyDividedRegion(std::vector<std::pair<unsigned long, unsigned long> > &justDividedCellsRN2, unsigned long testCell)
{
    bool found = false;
    std::vector<RNPairType>::iterator iteJustDivRN;
    iteJustDivRN = find_if(justDividedCellsRN2.begin(), justDividedCellsRN2.end(), IsEqualLong(testCell));
    if (iteJustDivRN != justDividedCellsRN2.end())
    {
        found = true;
    }
    return found;
}

/* True if a region (RN) belong to the first layer of cell
* @param neighbourMap the list of neighbor region for each region (RN)
* @param borderCells2 the list of border region (RN)
* @param testCell the RN to search for
* @return true if testCell has a neighbor that is listed as bordercell
*/
bool IsFirstLayerCell(std::map<unsigned long, std::vector<unsigned long> > &neighbourMap, std::vector<unsigned long> &borderCells2, unsigned long testCell)
{
    bool found = false;
    std::vector<unsigned long>::iterator iteNeighbor;
    for (iteNeighbor = neighbourMap[testCell].begin(); iteNeighbor != neighbourMap[testCell].end(); ++iteNeighbor)
    {
        std::vector<unsigned long>::iterator iteBorder;
        iteBorder = find(borderCells2.begin(), borderCells2.end(), *iteNeighbor);
        if (iteBorder != borderCells2.end())
        {
            found = true;
            break;
        }
    }
    return found;
}

/* Replace a division pair by a new division pair in both RN and AN just divided cell list
* @param justDividedCellsRN2 the list of just divided region (RN)
* @param justDividedCellsAN2 the list of just divided tag (AN)
* @param oldDivision the division to replace < <RN1,RN2> ; <AN1,AN2> >
* @param newDivision the division to add < <RN1,RN2> ; <AN1,AN2> >
* @return true if all went fine
*/
bool ReplaceDivisionPair(std::vector<RNPairType> &justDividedCellsRN2, std::vector<ANPairType> &justDividedCellsAN2, std::pair<RNPairType, ANPairType> oldDivision, std::pair<RNPairType, ANPairType> newDivision)
{
    // parse the arguments for facilitate value handling
    RNPairType oldDivisionRN = oldDivision.first;
    ANPairType oldDivisionAN = oldDivision.second;
    RNPairType newDivisionRN = newDivision.first;
    ANPairType newDivisionAN = newDivision.second;

    // search for the old pair in the RN and AN list
    std::vector<RNPairType>::iterator iteOldRN = find(justDividedCellsRN2.begin(), justDividedCellsRN2.end(), oldDivisionRN);
    std::vector<ANPairType>::iterator iteOldAN = find(justDividedCellsAN2.begin(), justDividedCellsAN2.end(), oldDivisionAN);

    // verification
    if (iteOldRN == justDividedCellsRN2.end())
    {
        oldDivisionRN = SwitchPair(oldDivisionRN);
        iteOldRN = find(justDividedCellsRN2.begin(), justDividedCellsRN2.end(), oldDivisionRN);
        if (iteOldRN == justDividedCellsRN2.end())
        {
            std::cerr << "Error 'ReplaceDivisionPair' : could not find the division [" << oldDivisionRN.first << " , " << oldDivisionRN.second << "] in the RN division list" << std::endl;
            std::cerr << "we did not found " << oldDivisionRN.first << ";" << oldDivisionRN.second << std::endl;
            return EXIT_FAILURE;
        }
    }
    if (iteOldAN == justDividedCellsAN2.end())
    {
        oldDivisionAN = SwitchPair(oldDivisionAN);
        iteOldAN = find(justDividedCellsAN2.begin(), justDividedCellsAN2.end(), oldDivisionAN);
        if (iteOldAN == justDividedCellsAN2.end())
        {
            std::cerr << "Error 'ReplaceDivisionPair' : could not find the division [" << oldDivisionAN.first << " , " << oldDivisionAN.second << "] in the AN division list" << std::endl;
            std::cerr << "we did not found " << oldDivisionAN.first << ";" << oldDivisionAN.second << std::endl;
            return EXIT_FAILURE;
        }
    }
    if ((iteOldRN - justDividedCellsRN2.begin()) != (iteOldAN - justDividedCellsAN2.begin()))
    {
        std::cerr << "Error 'ReplaceDivisionPair' : Inconsistancy in the AN and RN division list ... " << std::endl;
        return EXIT_FAILURE;
    }

    // replace old division with the new division
    replace(justDividedCellsRN2.begin(), justDividedCellsRN2.end(), oldDivisionRN, newDivisionRN);
    replace(justDividedCellsAN2.begin(), justDividedCellsAN2.end(), oldDivisionAN, newDivisionAN);

    return EXIT_SUCCESS;
}