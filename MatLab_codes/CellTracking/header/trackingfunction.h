#ifndef __trackingfunction_h__
#define __trackingfunction_h__

#include "trackingincludes.h"

/* Add a neighbour in the neighbour map
* @param neighbourMap a pointer to the neighbour map
* @param currentCell the cell RN
* @param neighbour1 the first neighbour RN
* @param neighbour2 the second neighbour RN (optional)
* @param neighbour3 the third neighbour RN (optional)
*/
void AddNeighboursInMap(std::map<unsigned long, std::vector<unsigned long> > *, unsigned long, unsigned long, unsigned long, unsigned long);

/* Fill the neighbour map and print the skeleton pixels info file
* @param labeledITKimage a pointer to the labeled image
* @param neighbourMap a pointer to the map to fill
* @param outputfilename the name of the output file
* @param bordercells a reference to the RN list of the border cells
*/
void FillNeighboursList(const ITKImageType::Pointer &, std::map<unsigned long, std::vector<unsigned long> > *, std::string, std::vector<unsigned long> &);

/* Print a neighbour map
* @param filename the output file name
* @param neighbourMap the map to print
*/
void PrintNeighboursMap(std::string, const std::map<unsigned long, std::vector<unsigned long> > &);

/* Read border cells file contents and store it in a vector.
* @param filename the file name
* @param bordercells the destination vector
*/
void ReadBorderCellsFile(std::string, std::vector<unsigned long> &);

/* Print border cells in file.
* @param filename the file name
* @param bordercells the destination vector
*/
void PrintBordercells(std::string, const std::vector<unsigned long> &);

/* Find Minimum in an array
* @param tab the array
* @param nbElement the number of elements in the array
* @return the index of the Minimum in the array
*/
unsigned int Minimum(const double *, unsigned int);

/* Find the index of the minimum in an array
* @param tab the array
* @return the index of the Minimum in the array
*/
unsigned int MinimumIndex(const std::vector<double>);

/* Sort couple by number
* @param couple1 the first couple
* @param couple2 the second couple
* @return true if couple1 < couple2
*/
bool CompareByNumber(Couple, Couple);

/* Shift coordonate point (x,y) according to PIV (u,v)
* @param cIndex the coordinate index to be shifted
* @param itePIV1 the iterator on the PIV_U
* @param itePIV2 the iterator on the PIV_V
* @param imageSize the image size
* @param bias additionnal bias
* @return new index
*/
ITKImageType::IndexType ShiftPointByPIV(ITKImageType::IndexType &, IteratorType &, IteratorType &, ITKImageType::SizeType &, bool);

/* Calculate cell overlap between  previous and current frames
* @param overlap the overlap map to be calculate
* @param overlapFiller the overlap data structure
* @param iteImage1 the iterator on the first frame
* @param iteImage2 the iterator on the second frame
* @param itePIV1 the iterator on the PIV_U
* @param itePIV2 the iterator on the PIV_V
* @param imageSize the image size
*/
void OverlapCell2CellComparison(std::map<int, std::vector<Couple> > &, Couple &, ITKImageType *, ITKImageType *, ITKImageType *, ITKImageType *, ITKImageType::SizeType &);

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
void CentroidToCellOverlap(std::pair<double, double> *, unsigned long, std::map<int, std::vector<Couple> > &, GeometryFilterType *, GeometryFilterType *, IteratorType &, IteratorType &, IteratorType &, IteratorType &, ITKImageType::SizeType &);

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
//bool DivisionVerificationFunction(unsigned long, unsigned long, unsigned long, std::map<unsigned long, std::vector<unsigned long> > &, std::map<unsigned long, double> &, GeometryFilterType *, GeometryFilterType *, std::vector<unsigned long> &, std::vector<unsigned long> &, double, double, double, bool, std::ofstream &);
bool DivisionVerificationFunction(unsigned long, unsigned long, unsigned long, std::map<int, std::map<int, std::vector<CorrespondanceCouple> > > &, std::map<unsigned long, std::vector<unsigned long> > &, std::map<unsigned long, double> &, GeometryFilterType *, GeometryFilterType *, std::vector<unsigned long> &, std::vector<unsigned long> &, double, double, double, unsigned int, unsigned int, bool, std::ofstream &);

/* True if a region (RN) is listed as newly coalesced elements
* NB: a region (RN) is newly coalesced if it containes at least one newly coalesced tag (AN)
* @param newlyCoalescedElements the map of newly coalesced region and tag
* @param testCell the RN to search for
* @return true if found
*/
bool IsCoalesced(std::map<unsigned long, std::vector<std::string> > &, unsigned long);

/* True if a region (RN) is listed as just divided cell
* @param justDividedCellsRN2 the list of newly divided region (RN)
* @param testCell the RN to search for
* @return true if found
*/
bool IsNewlyDividedRegion(std::vector<std::pair<unsigned long, unsigned long> > &, unsigned long);

/* True if a region (RN) belong to the first layer of cell
* @param neighbourMap the list of neighbor region for each region (RN)
* @param borderCells2 the list of border region (RN)
* @param testCell the RN to search for
* @return true if testCell has a neighbor that is listed as bordercell
*/
bool IsFirstLayerCell(std::map<unsigned long, std::vector<unsigned long> > &, std::vector<unsigned long> &, unsigned long);

/* Replace a division pair by a new division pair in both RN and AN just divided cell list
* @param justDividedCellsRN2 the list of just divided region (RN)
* @param justDividedCellsAN2 the list of just divided tag (AN)
* @param oldDivision the division to replace < <RN1,RN2> ; <AN1,AN2> >
* @param newDivision the division to add < <RN1,RN2> ; <AN1,AN2> >
* @return true if all went fine
*/
bool ReplaceDivisionPair(std::vector<RNPairType> &, std::vector<ANPairType> &, std::pair<RNPairType, ANPairType>, std::pair<RNPairType, ANPairType>);


#endif // __trackingfunction_h__