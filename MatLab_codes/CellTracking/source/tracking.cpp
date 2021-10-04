/*** TRACKING.cxx
* cell tracking main program
*
* @version 4.0.1
* WARNING: WHEN CHANGING VERSION NUMBER,REMEMBER TO UPDATE string "tracking_version" in the file trackingincludes.h
* @author Raphael MARCHAND & Boris GUIRAO & Yûki GOYA & Stephane RIGAUD
* @date 2016-01-19
*
* NB centroidmatlab = centroiditk+1 puisque en itk les pixels sont numérotés à partir de zéro
*
* NB notion de pointeur importante ici :
* Les filtres itk sont des pointeurs donc si on fait image2=image1
* et qu'ensuite, on charge une autre image dans image1, les deux images image1 et image2
* contiennent la même adresse de l'image qu'on vient de charger, car image2=image1 fait que les
* deux pointeurs pointent sur le même objet.
*
* NB sur la nécessité d'utiliser ->Delete() pour labelgeometryimagefilter :
* pour connectedcomponentimagefilter, imagefilereader, rescaleintensityimagefilter, imagefilewriter on peut en déclarer un seul,
* et modifier plusieurs fois l'input (en modifiant le reader avec set filename)
* En revanche pour labelgeometryimagefilter, il faut le détruire avec ->Delete() et le recréer à chaque fois si on veut changer l'image sur lequel
* il s'applique
*
* couple peut être appelé cell_j_data
*
*
* ** Additional commentary from Stephane Rigaud (2016-01-07) **
*
* The base program structure has change: 
* - The file trackingincludes.h contains now all the necessary includes, the variable type definition, and a few templated function.
* - The file tracking.cpp is now the main program file and only contains the main() and the 5 correcting patches function (correspondance patch, blue patch, red-green patch, apopthosis patch, red-green ext patch)
* - The files trackingfunction(h/cpp) containes the source and header for several smaller function used in the main function and/or in the patches.
*
* The tracking is based on a set a variables that store all the information frame by frame:
* - correspondance - it is a map< frame, map< RN, vector<AN> > >, it is the main variable of the tracking and contains, for each RN of each frame, the list of AN associated. In a perfect world this list is only made of 1 value.
* - reversetrack - it is a list< frame, AN, RN >, it is build at the same time as correspondance and is it reverse. For each AN of each frame, it gives you its corresponding RN. (currently only usefull in Apopthosis patch)
* - justDividedCellsAN and justDividedCellsRN are created at each iteration and simply store the list of daughter cells that appear at this frame. 
* - newlyCoalescedCells is created at each iteration and containes the list of RN and one or more AN that just coalesced. Careful, an old coalesced region can become newly coalesced if it assimilate at least one new AN, in this case, the newlyCoalescedCells will only list the new coalesced AN, not the old AN.
*
* Those variables must be correctly managed as they are critical for the good behaviour of the global program. A bad handly of them will lead to a SegFault.
*
*/

#include "trackingincludes.h"
#include "trackingfunction.h"

#ifndef __DATE__
#define __DATE__ "N/A"
#define __TIME__ "N/A"
#endif


// ************************************************************************************************************************************************************************************************* \\

/* Search for each blue cell (new) a red cell (coalesced) in its neighborhood
* if we found a blue/red couple, we replace the blue with one of the red tag
* @param newlyCoalescedElements the list of newly coalesced region (RN)
* @param correspondance the list of correspondance
* @param neighbourMap the list of neighboor for each region (RN)
* @param justDividedCellsRN2 the just divided list (RN)
* @param justDividedCellsAN2 the just divided list (AN)
* @param borderCells2 the list of border cell
* @param newCellsRN_current the list of new cell (RN)
* @param newCellsAN_current the list of new cell (AN)
* @param reverseTrack the reverse correspondance
* @param currentFrame the current frame
* @param firstImage the first image
* @param outputPath the output tracking path
* @param debugMode the debug mode flag
*/
void
BlueCorrectionPatch(std::map<unsigned long, std::vector<std::string> > &newlyCoalescedElements, std::map<int, std::map<int, std::vector<CorrespondanceCouple> > > &correspondance, std::map<unsigned long, std::vector<unsigned long> > &neighbourMap, std::vector<std::pair< unsigned long, unsigned long> > &justDividedCellsRN2, std::vector<std::pair< std::string, std::string> > &justDividedCellsAN2, std::vector<unsigned long> &borderCells2, std::vector<unsigned long> &newCellsRN_current, std::vector<unsigned long> &newCellsAN_current, std::map<std::string, unsigned long> *reverseTrack, unsigned int currentFrame, unsigned int firstImage, std::string outputPath, bool debugMode)
{
    // patch result file
    std::ofstream rbSolved((outputPath + "/Solved_B_" + intToString(firstImage + currentFrame - 1) + ".txt").c_str());
    std::ofstream blog;
    if (debugMode)
    {
        blog.open((outputPath + "/debugResolved_B_" + intToString(firstImage + currentFrame - 1) + ".txt").c_str());
    }

    bool mainSolved;
    std::vector<unsigned long>::iterator iteNewCellRNs;
    std::vector<unsigned long>::iterator iteNewCellANs;
    for (iteNewCellRNs = newCellsRN_current.begin(), iteNewCellANs = newCellsAN_current.begin(); iteNewCellRNs != newCellsRN_current.end();)
    {
        mainSolved = false;

        if (debugMode)
        {
            blog << "/*---------------------------------------------------------------------*/" << std::endl;
            blog << "trying to solve the new cell " << *iteNewCellRNs << " (" << correspondance[currentFrame][*iteNewCellRNs].front().getAbsoluteTag() << ")" << std::endl;
        }

        std::pair<unsigned int, std::string> currentNewCell = std::make_pair(*iteNewCellRNs, correspondance[currentFrame][*iteNewCellRNs].front().getAbsoluteTag());

        // look for cell event in the 1-neighbourhood of the currentNewCell
        std::vector<unsigned long> newRedNeighbor;
        std::vector<unsigned long> newGreenNeighbor;
        std::vector<unsigned long>::iterator iteCellEvent;
        for (iteCellEvent = neighbourMap[currentNewCell.first].begin(); iteCellEvent != neighbourMap[currentNewCell.first].end(); ++iteCellEvent)
        {
            bool notBorderCell = find(borderCells2.begin(), borderCells2.end(), *iteCellEvent) == borderCells2.end();
            if (notBorderCell)
            {
                bool redFound(false), greenFound(false);
                std::pair<unsigned long, std::string> currentNeighbor = std::make_pair(*iteCellEvent, correspondance[currentFrame][*iteCellEvent].front().getAbsoluteTag());

                if (debugMode)
                {
                    blog << "\tthe neighbor cell " << currentNeighbor.first << "(" << currentNeighbor.second << ") -> ";
                }

                if (IsNewlyDividedRegion(justDividedCellsRN2, currentNeighbor.first))  // is the neighbor a newly divided cell ?
                {
                    newGreenNeighbor.push_back(currentNeighbor.first);
                    greenFound = true;
                    if (debugMode)
                    {
                        blog << "is newly divided";
                    }
                }
                if (IsCoalesced(newlyCoalescedElements, currentNeighbor.first)) // is the neighbor a newly coalesed cell ?
                {
                    newRedNeighbor.push_back(currentNeighbor.first);
                    redFound = true;
                    if (debugMode)
                    {
                        if (greenFound)
                        {
                            blog << " && ";
                        }
                        blog << "is coalesced";
                    }
                }
                if (!redFound && !greenFound)
                {
                    if (debugMode)
                    {
                        blog << "is nothing";
                    }
                }
                if (debugMode)
                {
                    blog << std::endl;
                }
            }
        } // end loop for neighborhood

        // 3 cases
        // 1 - it is a first layer cell -> we do nothing
        // 2 - it has a coalesced cell next to her -> we solve it
        // 3 - it has a divided cell next to her -> we extend our search

        bool firstLayerFlag = IsFirstLayerCell(neighbourMap, borderCells2, currentNewCell.first);
        if (firstLayerFlag)
        {
            if (debugMode)
            {
                blog << "It's a first layer cell, we do nothing. Bye bye !" << std::endl;
            }
        } // end case 1
        else if (!newRedNeighbor.empty())
        {
            if (debugMode)
            {
                blog << "We have found " << newRedNeighbor.size() << " coalesced regions in the new cell neighborhood" << std::endl;
            }

            std::vector<double>                          distances;
            std::vector<ANPairType>                      pairingCell;
            std::vector<RNPairType>                      pairingRegion;
            std::vector<std::pair< RNPairType, ANPairType > > pairingDivided;

            std::vector<unsigned long>::iterator iteRedNeighbor;
            for (iteRedNeighbor = newRedNeighbor.begin(); iteRedNeighbor != newRedNeighbor.end(); ++iteRedNeighbor)  // for each coalesced in 1-neighborhood
            {
                if (debugMode)
                {
                    blog << "\tfor the coalesced region #" << *iteRedNeighbor << " ... " << std::endl;
                }

                unsigned long coalescedRN = *iteRedNeighbor;
                std::vector<CorrespondanceCouple>::iterator iteCorrespondance;
                for (iteCorrespondance = correspondance[currentFrame][coalescedRN].begin();
                    iteCorrespondance != correspondance[currentFrame][coalescedRN].end(); iteCorrespondance++)  // for each tag in the coalesced
                {
                    // coalesced cell centroid
                    double redX = iteCorrespondance->GetCentroid(0);  // coalesced centroid X
                    double redY = iteCorrespondance->GetCentroid(1);  // coalesced centroid Y

                    // fake new cell centroid
                    double blueX = correspondance[currentFrame][currentNewCell.first].front().GetCentroid(0);  // new cell centroid X
                    double blueY = correspondance[currentFrame][currentNewCell.first].front().GetCentroid(1);  // new cell centroid Y

                    // pairing (RN/AN)
                    std::string tmpAN = correspondance[currentFrame][currentNewCell.first].front().getAbsoluteTag();
                    pairingCell.push_back(std::make_pair(iteCorrespondance->getAbsoluteTag(), tmpAN));                   // pairing cell AN (coalesced/new)
                    pairingRegion.push_back(std::make_pair(coalescedRN, currentNewCell.first));                          // pairing region RN (coalesced/new)
                    distances.push_back((blueX - redX) * (blueX - redX) + (blueY - redY) * (blueY - redY));         // pairing distance

                    // pairing div (region,cell)
                    std::vector<ANPairType>::iterator iteSearchDivAN = find_if(justDividedCellsAN2.begin(), justDividedCellsAN2.end(), IsEqualStr(pairingCell.back().first));
                    std::vector<RNPairType>::iterator iteSearchDivRN = justDividedCellsRN2.begin() + (iteSearchDivAN - justDividedCellsAN2.begin());

                    RNPairType divRegion;
                    ANPairType divCell;
                    bool divisionFound = false;
                    if (iteSearchDivAN != justDividedCellsAN2.end() && iteSearchDivRN != justDividedCellsRN2.end())
                    {
                        divRegion = *iteSearchDivRN;
                        divCell = *iteSearchDivAN;
                        divisionFound = true;
                    }
                    else
                    {
                        divRegion = std::make_pair(0, 0);
                        divCell = std::make_pair("0", "0");
                    }
                    pairingDivided.push_back(std::make_pair(divRegion, divCell));

                    if (debugMode)
                    {
                        blog << "\tdistance coalesced cell " << pairingRegion.back().first << "(" << pairingCell.back().first << ")" << " - ";
                        blog << "fake new cell: " << pairingRegion.back().second << "(" << pairingCell.back().second << ") = " << distances.back() << std::endl;
                        if (divisionFound)
                        {
                            blog << "\t\tdetected division tag : " << pairingDivided.back().first.first << "," << pairingDivided.back().first.second
                                << " (" << pairingDivided.back().second.first << "," << pairingDivided.back().second.second << ")." << std::endl;
                        }
                    }
                } // end loop for each coalesced tag
            } // end loop for each coalesced region found

            // get Minimum distance
            unsigned int minindix = MinimumIndex(distances);
            if (debugMode)
            {
                blog << "solve : coalesced cell " << pairingRegion[minindix].first << "(" << pairingCell[minindix].first << ") ";
                blog << "in the new cell " << pairingRegion[minindix].second << "(" << pairingCell[minindix].second << ")" << std::endl;
            }

            // update correspondance
            std::vector<CorrespondanceCouple>::iterator iteCorrespondance;
            for (iteCorrespondance = correspondance[currentFrame][pairingRegion[minindix].first].begin();
                iteCorrespondance != correspondance[currentFrame][pairingRegion[minindix].first].end(); ++iteCorrespondance)
            {
                if (iteCorrespondance->getAbsoluteTag().compare(pairingCell[minindix].first) == 0)
                {
                    if (debugMode)
                    {
                        blog << "\t\tkill the new cell [" << correspondance[currentFrame][pairingRegion[minindix].second].front().getAbsoluteTag() << "] from region #" << pairingRegion[minindix].second << std::endl;
                    }
                    reverseTrack[currentFrame - 1][correspondance[currentFrame][pairingRegion[minindix].second].front().getAbsoluteTag()] = 0; // del new cell
                    correspondance[currentFrame][pairingRegion[minindix].second].clear();                                                      // clear newcell

                    if (debugMode)
                    {
                        blog << "\t\tmove the coalesced tag [" << iteCorrespondance->getAbsoluteTag() << "] from region #" << pairingRegion[minindix].first;
                        blog << " to the region #" << pairingRegion[minindix].second << std::endl;
                    }
                    reverseTrack[currentFrame - 1][iteCorrespondance->getAbsoluteTag()] = pairingRegion[minindix].second;  // update coalesced cell
                    correspondance[currentFrame][pairingRegion[minindix].second].push_back(*iteCorrespondance);            // add the newly coalesced cell in the new cell
                    correspondance[currentFrame][pairingRegion[minindix].first].erase(iteCorrespondance);                  // delete newly coalesced element

                    if (find_if(justDividedCellsAN2.begin(), justDividedCellsAN2.end(), IsEqualStr(pairingCell[minindix].first)) != justDividedCellsAN2.end())
                    {
                        if (debugMode)
                        {
                            blog << "\twe are messing around with a just divided cell, we need to update the list." << std::endl;
                        }

                        // create div pair to replace in the list
                        RNPairType oldDivRN = pairingDivided[minindix].first;
                        ANPairType oldDivAN = pairingDivided[minindix].second;
                        if (debugMode)
                        {
                            blog << "\t\tthe old division : " << oldDivRN.first << "," << oldDivRN.second << " (" << oldDivAN.first << "," << oldDivAN.second << ")" << std::endl;
                        }

                        // check order of the pair in order to keep the 1-2 daughter order
                        RNPairType newDivRN = std::make_pair(oldDivRN.first, pairingRegion[minindix].second);
                        if (pairingRegion[minindix].first == oldDivRN.first)
                        {
                            newDivRN = std::make_pair(pairingRegion[minindix].second, oldDivRN.second);
                        }

                        // we update the list
                        ReplaceDivisionPair(justDividedCellsRN2, justDividedCellsAN2, std::make_pair(oldDivRN, oldDivAN), std::make_pair(newDivRN, oldDivAN));

                        if (debugMode)
                        {
                            blog << "\t\tupdate the division region :\t" << "[" << oldDivRN.first << "," << oldDivRN.second << "] by the new [" << newDivRN.first << "," << newDivRN.second << "]" << std::endl;
                            blog << "\t\tequivalent in AN :\t" << "[" << oldDivAN.first << "," << oldDivAN.second << "] by the new [" << oldDivAN.first << "," << oldDivAN.second << "]" << std::endl;
                        }
                    } // end if coalesced tag is just divided

                    // writing patch changes in RN numbers, from coalesced to new cell
                    rbSolved << pairingRegion[minindix].first << " " << pairingRegion[minindix].second << std::endl;

                    // we clean coalesced list, raise the flag, and get out of the loop, we are done here
                    std::vector<std::string>::iterator iteToClean;
                    iteToClean = find(newlyCoalescedElements[pairingRegion[minindix].first].begin(), newlyCoalescedElements[pairingRegion[minindix].first].end(), pairingCell[minindix].first);
                    if (debugMode)
                    {
                        blog << "Is the region #" << pairingRegion[minindix].first << " still coalesced ? " << std::endl;
                    }
                    if (correspondance[currentFrame][pairingRegion[minindix].first].size() == 1)
                    {
                        if (debugMode)
                        {
                            blog << "\tOnly 1 tag left in the region, it not a coalesced anymore ..." << std::endl;
                        }
                        std::map<unsigned long, std::vector<std::string> >::iterator iteNewCoalescedCleaner;
                        iteNewCoalescedCleaner = newlyCoalescedElements.find(pairingRegion[minindix].first);
                        newlyCoalescedElements.erase(iteNewCoalescedCleaner);

                    }
                    else if (iteToClean != newlyCoalescedElements[pairingRegion[minindix].first].end())
                    {
                        newlyCoalescedElements[pairingRegion[minindix].first].erase(iteToClean);
                        if (newlyCoalescedElements[pairingRegion[minindix].first].empty())
                        {
                            if (debugMode)
                            {
                                blog << "\tOnly old tag left in the region, we stop there ..." << std::endl;
                            }
                            std::map<unsigned long, std::vector<std::string> >::iterator iteNewCoalescedCleaner;
                            iteNewCoalescedCleaner = newlyCoalescedElements.find(pairingRegion[minindix].first);
                            newlyCoalescedElements.erase(iteNewCoalescedCleaner);
                        }
                    }
                    else
                    {
                        if (debugMode)
                        {
                            blog << "\tIt still contain at least 2 tag (and one of them is newlyCoalesced)" << std::endl;
                        }
                    }

                    mainSolved = true;
                    break;
                } // end if the coalesced tag correspond to the one we want
            } // end loop for each coalesced tag in the region
        } // end case 2
        else if (newRedNeighbor.empty() && !newGreenNeighbor.empty())
        {
            if (debugMode)
            {
                blog << "No Coalesced cells detected, but " << newGreenNeighbor.size() << " just divided ... we go to the 2-neighborhood." << std::endl;
            }

            // calculate pairing for each newlydividedcell in the neighborhood and each of their coalesced 2-neighbors
            std::vector<double>                          distances;
            std::vector<ANPairType>                      pairingCell;
            std::vector<RNPairType>                      pairingRegion;
            std::vector<std::pair< RNPairType, ANPairType > > pairingDivided;

            // for each divided cell in the 1-neighborhood
            std::vector<unsigned long>::iterator iteGreenNeighbor;
            for (iteGreenNeighbor = newGreenNeighbor.begin(); !mainSolved && iteGreenNeighbor != newGreenNeighbor.end(); ++iteGreenNeighbor)
            {
                newRedNeighbor.clear();
                bool newCellFlag = false;
                std::pair<unsigned long, std::string> currentGreenCell = std::make_pair(*iteGreenNeighbor, correspondance[currentFrame][*iteGreenNeighbor].front().getAbsoluteTag());

                if (debugMode)
                {
                    blog << "... for the cell " << currentGreenCell.first << " (" << currentGreenCell.second << ") ... " << std::endl;
                }

                // for each coalesced cell in the 2-neighborhood
                std::vector<unsigned long>::iterator iteCellEvent;
                for (iteCellEvent = neighbourMap[currentGreenCell.first].begin(); iteCellEvent != neighbourMap[currentGreenCell.first].end(); ++iteCellEvent)
                {
                    std::pair<unsigned long, std::string> currentNeighbor;
                    bool notBorderCell = find(borderCells2.begin(), borderCells2.end(), *iteGreenNeighbor) == borderCells2.end();
                    if (notBorderCell)
                    {
                        currentNeighbor = std::make_pair(*iteCellEvent, correspondance[currentFrame][*iteCellEvent].front().getAbsoluteTag());
                        if (debugMode)
                        {
                            blog << "\t\tthe neighbor cell " << currentNeighbor.first << "(" << currentNeighbor.second << ") -> ";
                        }
                        std::vector<unsigned long>::iterator iteNewCellCheck;
                        iteNewCellCheck = find(newCellsRN_current.begin(), newCellsRN_current.end(), currentNeighbor.first);
                        if (iteNewCellCheck != newCellsRN_current.end())
                        {
                            newCellFlag = true;
                            if (debugMode)
                            {
                                blog << "is a new cell";
                            }
                        }
                        else if (IsCoalesced(newlyCoalescedElements, currentNeighbor.first)) // is the neighbor a newly coalesed cell ?
                        {
                            newRedNeighbor.push_back(currentNeighbor.first);
                            if (debugMode)
                            {
                                blog << "is coalesced";
                                if (IsNewlyDividedRegion(justDividedCellsRN2, currentNeighbor.first))  // is the neighbor a newly divided cell ?
                                {
                                    if (debugMode)
                                    {
                                        blog << " ( && is newly divided)";
                                    }
                                }
                            }
                        }
                        else
                        {
                            if (debugMode)
                            {
                                blog << "is nothing";
                                if (IsNewlyDividedRegion(justDividedCellsRN2, currentNeighbor.first))  // is the neighbor a newly divided cell ?
                                {
                                    if (debugMode)
                                    {
                                        blog << " ( && is newly divided)";
                                    }
                                }
                            }
                        }
                        if (debugMode)
                        {
                            blog << std::endl;
                        }
                    } // end if border cell
                } // end loop for each 2-neighbor

                if (!newRedNeighbor.empty() && !newCellFlag) // case 2 but in the 2-neighborhood
                {
                    if (debugMode)
                    {
                        blog << "\twe have detected " << newRedNeighbor.size() << " coalesced cell in the 2-neighborhood ..." << std::endl;
                    }

                    // for each coalesced cell detected in the neighborhood of the newlydividedcell
                    std::vector<unsigned long>::iterator iteRedNeighbor;
                    for (iteRedNeighbor = newRedNeighbor.begin(); iteRedNeighbor != newRedNeighbor.end(); ++iteRedNeighbor)  // for each coalesced in 1-neighborhood
                    {
                        if (debugMode)
                        {
                            blog << "\t\tfor the coalesced region #" << *iteRedNeighbor << " ... " << std::endl;
                        }
                        std::vector<CorrespondanceCouple>::iterator iteCorrespondance;
                        for (iteCorrespondance = correspondance[currentFrame][*iteRedNeighbor].begin();
                            iteCorrespondance != correspondance[currentFrame][*iteRedNeighbor].end(); iteCorrespondance++)  // for each tag in the coalesced
                        {
                            // coalesced cell centroid
                            double redX = iteCorrespondance->GetCentroid(0);  // coalesced centroid X
                            double redY = iteCorrespondance->GetCentroid(1);  // coalesced centroid Y

                            // fake new cell centroid
                            double blueX = correspondance[currentFrame][currentGreenCell.first].front().GetCentroid(0);   // new cell centroid X
                            double blueY = correspondance[currentFrame][currentGreenCell.first].front().GetCentroid(1);   // new cell centroid Y

                            // pairing (RN/AN)
                            std::string tmpAN = correspondance[currentFrame][currentGreenCell.first].front().getAbsoluteTag();
                            pairingCell.push_back(std::make_pair(iteCorrespondance->getAbsoluteTag(), tmpAN));                 // pairing cell AN (coalesced/div)
                            pairingRegion.push_back(std::make_pair(*iteRedNeighbor, currentGreenCell.first));                  // pairing region RN (coalesced/div)
                            distances.push_back((blueX - redX) * (blueX - redX) + (blueY - redY) * (blueY - redY));       // pairing distance

                            // pairing div (region,cell)
                            std::vector<ANPairType>::iterator iteSearchDivAN = find_if(justDividedCellsAN2.begin(), justDividedCellsAN2.end(), IsEqualStr(pairingCell.back().first));
                            std::vector<RNPairType>::iterator iteSearchDivRN = justDividedCellsRN2.begin() + (iteSearchDivAN - justDividedCellsAN2.begin());

                            RNPairType divRegion;
                            ANPairType divCell;
                            bool divisionFound = false;
                            if (iteSearchDivAN != justDividedCellsAN2.end() && iteSearchDivRN != justDividedCellsRN2.end())
                            {
                                divRegion = *iteSearchDivRN;
                                divCell = *iteSearchDivAN;
                                divisionFound = true;
                            }
                            else
                            {
                                divRegion = std::make_pair(0, 0);
                                divCell = std::make_pair("0", "0");
                            }
                            pairingDivided.push_back(std::make_pair(divRegion, divCell));

                            if (debugMode)
                            {
                                blog << "\tdistance coalesced cell " << pairingRegion.back().first << "(" << pairingCell.back().first << ")" << " - ";
                                blog << "new divided cell: " << pairingRegion.back().second << "(" << pairingCell.back().second << ") = " << distances.back() << std::endl;
                                if (divisionFound)
                                {
                                    blog << "\t\t\tdetected division tag : " << pairingDivided.back().first.first << "," << pairingDivided.back().first.second << " (" << pairingDivided.back().second.first << "," << pairingDivided.back().second.second << ")." << std::endl;
                                }
                            }
                        } // end loop for each tag in coalesced
                    } // end loop for each coalesced cell
                } // end case 2 extended
            } // end loop for each newly divided

            if (!distances.empty())
            {
                // get Minimum distance
                unsigned int minindix = MinimumIndex(distances);
                if (debugMode)
                {
                    blog << "solve : coalesced cell " << pairingRegion[minindix].first << "(" << pairingCell[minindix].first << ") ";
                    blog << "-> the dividing cell " << pairingRegion[minindix].second << "(" << pairingCell[minindix].second << ") ";
                    blog << "-> the new cell " << currentNewCell.first << "(" << currentNewCell.second << ")" << std::endl;
                }

                // update correspondance
                std::vector<CorrespondanceCouple>::iterator iteCorrespondance;
                for (iteCorrespondance = correspondance[currentFrame][pairingRegion[minindix].first].begin();
                    iteCorrespondance != correspondance[currentFrame][pairingRegion[minindix].first].end(); ++iteCorrespondance)
                {
                    if (iteCorrespondance->getAbsoluteTag().compare(pairingCell[minindix].first) == 0)
                    {
                        // we kill the new cell
                        if (debugMode)
                        {
                            blog << "\t\tkill the new cell [" << currentNewCell.second << "] from region #" << currentNewCell.first << std::endl;
                        }
                        reverseTrack[currentFrame - 1][correspondance[currentFrame][currentNewCell.first].front().getAbsoluteTag()] = 0; // del new cell
                        correspondance[currentFrame][currentNewCell.first].clear();                                                      // clear newcell

                        // we move the divided cell in the new cell
                        if (debugMode)
                        {
                            blog << "\t\tmove the divided tag [" << pairingCell[minindix].second << "] from region #" << pairingRegion[minindix].second;
                            blog << " to the new cell region #" << currentNewCell.first << std::endl;
                        }
                        reverseTrack[currentFrame - 1][pairingCell[minindix].second] = currentNewCell.first;    // update divided cell
                        CorrespondanceCouple dividedCell = correspondance[currentFrame][pairingRegion[minindix].second].front();
                        correspondance[currentFrame][currentNewCell.first].push_back(dividedCell);              // add the newly divided cell in the new cell
                        correspondance[currentFrame][pairingRegion[minindix].second].clear();                   // delete newly divided element

                        // we update the division
                        if (find_if(justDividedCellsAN2.begin(), justDividedCellsAN2.end(), IsEqualStr(pairingCell[minindix].second)) != justDividedCellsAN2.end())
                        {
                            if (debugMode)
                            {
                                blog << "\t\t\twe need to update the new division list." << std::endl;
                            }

                            // create div pair to replace in the list
                            std::vector<ANPairType>::iterator iteJustDivAN = find_if(justDividedCellsAN2.begin(), justDividedCellsAN2.end(), IsEqualStr(pairingCell[minindix].second));
                            std::vector<RNPairType>::iterator iteJustDivRN = justDividedCellsRN2.begin() + (iteJustDivAN - justDividedCellsAN2.begin());
                            RNPairType oldDivRN = *iteJustDivRN;
                            ANPairType oldDivAN = *iteJustDivAN;
                            RNPairType newDivRN;
                            ANPairType newDivAN = oldDivAN;
                            if (debugMode)
                            {
                                blog << "\t\t\tthe old division : " << oldDivRN.first << "," << oldDivRN.second << " (" << oldDivAN.first << "," << oldDivAN.second << ")" << std::endl;
                            }

                            newDivAN = oldDivAN;
                            if (oldDivAN.first.compare(pairingCell[minindix].first) || oldDivAN.second.compare(pairingCell[minindix].first))
                            {
                                // the coalesced is in the division
                                if (oldDivAN.first.compare(pairingCell[minindix].first))
                                {   // (1/2) -> (2/3)
                                    // (C/D) -> (D/N) 
                                    newDivRN = std::make_pair(oldDivRN.second, currentNewCell.first);
                                    newDivAN = SwitchPair(oldDivAN);
                                }
                                else if (oldDivAN.second.compare(pairingCell[minindix].first))
                                {   // (2/1) -> (3/2)
                                    // (D/C) -> (N/D) 
                                    newDivRN = std::make_pair(currentNewCell.first, oldDivRN.first);
                                }
                            }
                            else
                            {
                                // (2/1) -> (3/2)
                                // (D/B) -> (N/B) 
                                // check order of the pair in order to keep the 1-2 daughter order
                                newDivRN = std::make_pair(oldDivRN.first, currentNewCell.first);
                                if (pairingRegion[minindix].second == oldDivRN.first)
                                {
                                    newDivRN = std::make_pair(currentNewCell.first, oldDivRN.second);
                                }
                            }
                            ReplaceDivisionPair(justDividedCellsRN2, justDividedCellsAN2, std::make_pair(oldDivRN, oldDivAN), std::make_pair(newDivRN, newDivAN));

                            if (debugMode)
                            {
                                blog << "\t\tupdate the division region :\t" << "[" << oldDivRN.first << "," << oldDivRN.second << "] by the new [" << newDivRN.first << "," << newDivRN.second << "]" << std::endl;
                                blog << "\t\tequivalent in AN :\t" << "[" << oldDivAN.first << "," << oldDivAN.second << "] by the new [" << newDivAN.first << "," << newDivAN.second << "]" << std::endl;
                            }
                        } // end if coalesced tag is just divided
                        else
                        {
                            if (debugMode)
                            {
                                std::cout << "blue patch case 3 - we did not update the list for the div cell ... THIS IS A MISTAKE!" << std::endl;
                            }
                        }

                        // we move the coalesced tag in the divided cell
                        if (debugMode)
                        {
                            blog << "\t\tmove the coalesced tag [" << iteCorrespondance->getAbsoluteTag() << "] from region #" << pairingRegion[minindix].first;
                            blog << " to the region #" << pairingRegion[minindix].second << std::endl;
                        }
                        reverseTrack[currentFrame - 1][iteCorrespondance->getAbsoluteTag()] = pairingRegion[minindix].second;  // update coalesced cell
                        correspondance[currentFrame][pairingRegion[minindix].second].push_back(*iteCorrespondance);            // add the newly coalesced cell in the new cell
                        correspondance[currentFrame][pairingRegion[minindix].first].erase(iteCorrespondance);                  // delete newly coalesced element

                        // we update if the coalesced is a division
                        std::vector<ANPairType>::iterator iteJustDivAN = find_if(justDividedCellsAN2.begin(), justDividedCellsAN2.end(), IsEqualStr(pairingCell[minindix].first));
                        if (iteJustDivAN != justDividedCellsAN2.end())
                        {
                            if (iteJustDivAN->first.compare(pairingCell[minindix].second) != 0 || iteJustDivAN->second.compare(pairingCell[minindix].second) != 0)
                            {
                                if (debugMode)
                                {
                                    blog << "\t\t\twe need to update the new division list because the coalesced tag is just divided." << std::endl;
                                }
                                // create div pair to replace in the list
                                RNPairType oldDivRN = pairingDivided[minindix].first;
                                ANPairType oldDivAN = pairingDivided[minindix].second;
                                if (debugMode)
                                {
                                    blog << "\t\t\tthe old division : " << oldDivRN.first << "," << oldDivRN.second << " (" << oldDivAN.first << "," << oldDivAN.second << ")" << std::endl;
                                }
                                // check order of the pair in order to keep the 1-2 daughter order
                                RNPairType newDivRN = std::make_pair(oldDivRN.first, currentNewCell.first);
                                if (pairingRegion[minindix].second == oldDivRN.first)
                                {
                                    newDivRN = std::make_pair(currentNewCell.first, oldDivRN.second);
                                }

                                // we update the list
                                ReplaceDivisionPair(justDividedCellsRN2, justDividedCellsAN2, std::make_pair(oldDivRN, oldDivAN), std::make_pair(newDivRN, oldDivAN));

                                if (debugMode)
                                {
                                    blog << "\t\t\tupdate the division region :" << "[" << oldDivRN.first << "," << oldDivRN.second << "] by the new [" << newDivRN.first << "," << newDivRN.second << "]" << std::endl;
                                    blog << "\t\t\tequivalent in AN :" << "[" << oldDivAN.first << "," << oldDivAN.second << "] by the new [" << oldDivAN.first << "," << oldDivAN.second << "]" << std::endl;
                                }
                            }
                        } // end if coalesced tag is just divided

                        // writing patch changes in RN numbers, from coalesced to new cell
                        rbSolved << pairingRegion[minindix].first << " " << pairingRegion[minindix].second << " " << currentNewCell.first << std::endl;

                        // we clean coalesced list, raise the flag, and get out of the loop, we are done here
                        std::vector<std::string>::iterator iteToClean;
                        iteToClean = find(newlyCoalescedElements[pairingRegion[minindix].first].begin(), newlyCoalescedElements[pairingRegion[minindix].first].end(), pairingCell[minindix].first);
                        if (debugMode)
                        {
                            blog << "Is the region #" << pairingRegion[minindix].first << " still coalesced ? " << std::endl;
                        }
                        if (correspondance[currentFrame][pairingRegion[minindix].first].size() == 1)
                        {
                            if (debugMode)
                            {
                                blog << "\tOnly 1 tag left in the region, it not a coalesced anymore ..." << std::endl;
                            }
                            std::map<unsigned long, std::vector<std::string> >::iterator iteNewCoalescedCleaner;
                            iteNewCoalescedCleaner = newlyCoalescedElements.find(pairingRegion[minindix].first);
                            newlyCoalescedElements.erase(iteNewCoalescedCleaner);

                        }
                        else if (iteToClean != newlyCoalescedElements[pairingRegion[minindix].first].end())
                        {
                            newlyCoalescedElements[pairingRegion[minindix].first].erase(iteToClean);
                            if (newlyCoalescedElements[pairingRegion[minindix].first].empty())
                            {
                                if (debugMode)
                                {
                                    blog << "\tOnly old tag left in the region, we stop there ..." << std::endl;
                                }
                                std::map<unsigned long, std::vector<std::string> >::iterator iteNewCoalescedCleaner;
                                iteNewCoalescedCleaner = newlyCoalescedElements.find(pairingRegion[minindix].first);
                                newlyCoalescedElements.erase(iteNewCoalescedCleaner);
                            }
                        }
                        else
                        {
                            if (debugMode)
                            {
                                blog << "\tIt still contain at least 2 tag (and one of them is newlyCoalesced)" << std::endl;
                            }
                        }

                        mainSolved = true;
                        break;
                    } // end if tag is the one we want
                } // end loop for each tag in coalesced
            } // end if not empty
        } // end case 3

        if (mainSolved)
        {
            iteNewCellRNs = newCellsRN_current.erase(iteNewCellRNs);
            iteNewCellANs = newCellsAN_current.erase(iteNewCellANs);
        }
        else
        {
            ++iteNewCellRNs;
            ++iteNewCellANs;
        }
        if (debugMode)
        {
            blog << std::endl;
        }
    } // end loop for each new cell
}

/* Detection of apoptosis
* @param newlyCoalescedElements the coalesced cell in the current frame
* @param reverseTrack the reverse correspondance track
* @param labelsAreas the label per compconn
* @param correspondance the correpondance map
* @param geometryFilter the geometry filter
* @param apopSizePencentil the percentil use for size threshold
* @param apoptosisCheckFrames the number of frame to look backwards
* @param apoptosisThreshold the size threshold
* @param currentFrame the frame index
* @param firstImage the first image index
* @param outputPath the output path
* @param debugMode the debug mode boolean
*/
void
ApoptosisPatch(std::map<unsigned long, std::vector<std::string> > &newlyCoalescedElements, std::map<std::string, unsigned long> *reverseTrack, std::map<unsigned long, unsigned long> *labelsAreas, std::map<int, std::map<int, std::vector<CorrespondanceCouple> > >  &correspondance, GeometryFilterType *geometryFilter, double apopSizePencentil, unsigned int apoptosisCheckFrames, double apoptosisThreshold, unsigned int currentFrame, unsigned int firstImage, std::string outputPath, bool debugMode)
{
    std::map<unsigned int, std::vector<std::pair<CorrespondanceCouple, double> > > apopMap;
    
    // patch result file
    std::ofstream aSolved((outputPath + "/Solved_A_" + intToString(firstImage + currentFrame - 1) + ".txt").c_str());
    std::ofstream debugApop;
    if (debugMode)
    {
        debugApop.open((outputPath + "/debugApoptosis_" + intToString(firstImage + currentFrame - 1) + ".txt").c_str());
    }

    // we calculate the Nth percentile of the cell size distribution at the current frame (4.0)
    std::vector<double> areaList;
    for (unsigned int i = 0; i <= geometryFilter->GetNumberOfObjects(); i++)
    {
        areaList.push_back(geometryFilter->GetVolume(i));
    }
    sort(areaList.begin(), areaList.end());
    double centileIndex = (apopSizePencentil / 100) * ceil(areaList.size() + 1);
    double centileResidual = fmod(centileIndex, 1) * (areaList[floor(centileIndex) + 1] - areaList[floor(centileIndex)]);
    double apopSizeThreshold = areaList[floor(centileIndex)] + centileResidual;

    // we check each newly coalesced region to see if one of the AN inside respect the apopthosis criteria
    std::map<unsigned long, std::vector<std::string> >::iterator iteNewCoalesced;
    for (iteNewCoalesced = newlyCoalescedElements.begin(); iteNewCoalesced != newlyCoalescedElements.end(); /*iteNewCoalesced++*/)
    {
        if (debugMode)
        {
            debugApop << "looking for delaminating cells in " << iteNewCoalesced->first << std::endl;
        }

        // does one of the AN inside the coalesced region correspond to an apoptosis
        std::vector<std::string>::iterator iteNC;
        for (iteNC = iteNewCoalesced->second.begin(); iteNC != iteNewCoalesced->second.end(); /*iteNC++*/)
        {
            // print cell RN [AN]
            if (debugMode)
            {
                debugApop << "the tag (" << *iteNC << ") in the region #" << iteNewCoalesced->first << std::endl;
            }

            bool deleteTag = false;                                          // delete AN centroids flag
            std::map<std::string, unsigned long>::iterator iteReverseTrack;  // iterator for reverse track
            std::vector<unsigned long> areas;                                // areas evolution

            // starting frame for mean area estimation
            int pFrame = currentFrame - apoptosisCheckFrames - 1;
            if (pFrame < 1)
            {
                pFrame = 1;
            }
            // we look for the cell area evolution in N previous frame
            for (; pFrame > 0; pFrame--) // reverse loop
            {
                if (debugMode)
                {
                    debugApop << "\t f=" << pFrame + firstImage - 1 << " ";
                }

                iteReverseTrack = reverseTrack[pFrame - 1].find(*iteNC); // search for the region in pFrame that correspond to the current AN
                if (iteReverseTrack != reverseTrack[pFrame - 1].end())   // we found a corresponding region
                {
                    if (correspondance[pFrame][iteReverseTrack->second].size() == 1) // if the region is not coalesced
                    {
                        areas.push_back(labelsAreas[pFrame - 1][iteReverseTrack->second]);
                        if (debugMode)
                        {
                            debugApop << "size=" << labelsAreas[pFrame - 1][iteReverseTrack->second] << " for (" << iteReverseTrack->second << ") ";
                        }
                    }
                    else // the region is coalesced, no area information
                    {
                        if (debugMode)
                        {
                            debugApop << "XC (" << iteReverseTrack->second << ")";
                        }
                        break;
                    }
                }
                else // we did not found a corresponding region
                {
                    if (debugMode)
                    {
                        debugApop << "XX";
                    }
                    break;
                }
            } // end loop for looking to cell area evolution

            // mean area computation
            if (areas.size() == 0)
            {
                if (debugMode)
                {
                    debugApop << std::endl;
                }
            }
            else
            {
                double meanRegArea = accumulate(areas.begin(), areas.end(), 0) / (double)areas.size();       // mean regular area
                double lastArea = labelsAreas[currentFrame - 2][reverseTrack[currentFrame - 2][*iteNC]];  // area evolution
                double areaEvolution = (lastArea - meanRegArea) / meanRegArea;
                if (debugMode)
                {
                    debugApop << "\t => f" << firstImage + currentFrame - 2 << " size=" << lastArea << " for (" << reverseTrack[currentFrame - 2][*iteNC] << ") variation = " << areaEvolution << std::endl;
                }
                if (areaEvolution < -apoptosisThreshold && lastArea < apopSizeThreshold)  // it is an apoptosis ?
                {

                    if (debugMode)
                    {
                        debugApop << "\tThe region # " << iteNewCoalesced->first << " has a tag " << *iteNC << " which delaminate in frame " << firstImage + currentFrame - 1 << std::endl;
                    }

                    // update the correspondance list
                    for (std::vector<CorrespondanceCouple>::iterator iteCorr = correspondance[currentFrame][iteNewCoalesced->first].begin();
                        iteCorr != correspondance[currentFrame][iteNewCoalesced->first].end(); iteCorr++)
                    {
                        if (iteCorr->getAbsoluteTag().compare(*iteNC) == 0)
                        {
                            if (debugMode)
                            {
                                debugApop << "\t\twe add the AN " << iteCorr->getAbsoluteTag() << " to the apopthosis map" << std::endl;
                            }
                            apopMap[iteNewCoalesced->first].push_back(std::make_pair(*iteCorr, areaEvolution));  // add the element to the list of apoptosis

                            if (debugMode)
                            {
                                debugApop << "\t\twe delete the AN " << iteCorr->getAbsoluteTag() << " from the region #" << iteNewCoalesced->first << std::endl;
                            }
                            reverseTrack[currentFrame - 1][iteCorr->getAbsoluteTag()] = 0;                       // update reverseTrack and set it to 0 (2012-11-15)
                            correspondance[currentFrame][iteNewCoalesced->first].erase(iteCorr);                 // remove the element from the correspondance list
                            deleteTag = true; // Apoptosis detected, we set the delete flag to true
                            break;
                        }
                    }
                }
                else
                {
                    if (debugMode)
                    {
                        debugApop << "\tThe region # " << iteNewCoalesced->first << " has a tag " << *iteNC << " does not delaminate in frame " << firstImage + currentFrame - 1 << std::endl;
                    }
                }
            } // end if area size

            if (deleteTag)
            {
                if (debugMode)
                {
                    debugApop << "we update the newlycoalescedlist by removing the tag " << *iteNC << " from the coalesced region #" << iteNewCoalesced->first << std::endl;
                }
                // remove from the newlycoalesced list, erase() return the incremented iterator
                iteNC = iteNewCoalesced->second.erase(iteNC);  
                aSolved << iteNewCoalesced->first << std::endl;
            }
            else
            {
                if (debugMode)
                {
                    debugApop << "no update needed, we go for the next tag ..." << std::endl;
                }
                iteNC++; // we increment the iterator
            }

            // refill correspondance if needed
            if (correspondance[currentFrame][iteNewCoalesced->first].size() == 0)
            {
                if (debugMode)
                {
                    debugApop << "\tRefilling correspondance ..." << std::endl;
                }
                std::vector<std::pair<CorrespondanceCouple, double> >::iterator iteKeep;
                std::vector<std::pair<CorrespondanceCouple, double> >::iterator iteApop;
                iteKeep = apopMap[iteNewCoalesced->first].begin(); // lucky looser init
                for (iteApop = apopMap[iteNewCoalesced->first].begin() + 1; iteApop != apopMap[iteNewCoalesced->first].end(); iteApop++)
                {
                    // keep the cell with highest ratio
                    if (iteApop->second > iteKeep->second)
                    {
                        iteKeep = iteApop;
                    }
                }
                std::string keepAN = iteKeep->first.getAbsoluteTag();
                correspondance[currentFrame][iteNewCoalesced->first].push_back(iteKeep->first);            // add it to the correspondance.
                reverseTrack[currentFrame - 1][iteKeep->first.getAbsoluteTag()] = iteNewCoalesced->first;  // update reverseTrack (2012-11-15)
                apopMap[iteNewCoalesced->first].erase(iteKeep);                                            // ... and remove the lucky looser from apoptosis map ...
                if (debugMode)
                {
                    debugApop << iteNewCoalesced->first << " [" << keepAN << "] lucky looser" << std::endl;
                }
            }

        } // end loop for on Inner Centroid List in Coalesced Cells

        if (correspondance[currentFrame][iteNewCoalesced->first].size() == 1 || iteNewCoalesced->second.empty())
        {
            if (debugMode)
            {
                debugApop << "The coalesced region #" << iteNewCoalesced->first << " is solved, we remove it from the newlycoalesced list." << std::endl;
            }
            iteNewCoalesced = newlyCoalescedElements.erase(iteNewCoalesced);
        }
        else
        {
            if (debugMode)
            {
                debugApop << "The coalesced region #" << iteNewCoalesced->first << " still contains some coalesced tag ... we move to the next one." << std::endl;
            }
            ++iteNewCoalesced;
        }

    } // end loop for on newlyCoalescedElements

    // close debugfile
    if (debugMode)
    {
        debugApop.close();
    }
    aSolved.close();

    // print the apoptosis list
    std::ofstream apopf((outputPath + "/delaminating_cells_" + intToString(firstImage + currentFrame - 2) + ".txt").c_str());
    if (!apopMap.empty())
    {
        std::map<unsigned int, std::vector<std::pair<CorrespondanceCouple, double> > >::iterator iteApopMap;
        for (iteApopMap = apopMap.begin(); iteApopMap != apopMap.end(); iteApopMap++)
        {
            std::vector<std::pair<CorrespondanceCouple, double> >::iterator iteInner;
            for (iteInner = iteApopMap->second.begin(); iteInner != iteApopMap->second.end(); iteInner++)
            {
                apopf << reverseTrack[currentFrame - 2][iteInner->first.getAbsoluteTag()] << std::endl;
            }
        }
    }
    else
    {
        apopf << -1 << std::endl; // if nothing detected
    }
    apopf.close();
}

/* Red-Green correstion patch
* @param newlyCoalescedElements the coalesced cell in the current frame
* @param correspondance the correpondance map
* @param reverseTrack the reverse correspondance track
* @param dividingCellsRN1 the dividing cells (mother RN)
* @param justDividedCellsRN2 the just divided cells (daughter RN)
* @param justDividedCellsAN2 the just divided cells (daughter AN)
* @param currentFrame the frame index
* @param firstImage the first image index
* @param outputPath the output path
* @param debugMode the debug mode boolean
*/
void
RedGreenCorrectionPatch(std::map<unsigned long, std::vector<std::string> > &newlyCoalescedElements, std::map<int, std::map<int, std::vector<CorrespondanceCouple> > > &correspondance, std::map<std::string, unsigned long> *reverseTrack, std::vector<unsigned long> &dividingCellsRN1, std::vector<RNPairType> &justDividedCellsRN2, std::vector<ANPairType> &justDividedCellsAN2, unsigned int currentFrame, unsigned int firstImage, std::string outputPath, bool debugMode)
{
    // list of solved cases
    std::ofstream rgsolved((outputPath + "/Solved_RG_" + intToString(firstImage + currentFrame - 1) + ".txt").c_str());
    std::ofstream rglog;
    if (debugMode)
    {
        rglog.open((outputPath + "/debugResolved_RG_" + intToString(firstImage + currentFrame - 1) + ".txt").c_str());
    }

    // for each newly coalesced region (a.k.a. a region, already coalesced or not, that just coalesced one or more new tag)
    std::map<unsigned long, std::vector<std::string> >::iterator iteNewCoalesced;
    for (iteNewCoalesced = newlyCoalescedElements.begin(); iteNewCoalesced != newlyCoalescedElements.end(); /*iteNewCoalesced++*/)
    {
        bool mainSolved = false;  // solved flag

        if (debugMode)
        {
            rglog << "/****************************************************************************/" << std::endl;
            rglog << "trying to solve coalescence of " << iteNewCoalesced->first << " (" << iteNewCoalesced->second.front();
            for (std::vector<std::string>::iterator iteCoalTag = iteNewCoalesced->second.begin() + 1; iteCoalTag != iteNewCoalesced->second.end(); ++iteCoalTag)
            {
                rglog << " | " << *iteCoalTag;
            }
            rglog << ")" << std::endl;
        }

        // we search if this region is also listed as just divided cell by looking in the justdividedcellRN structure if we find the region listed
        std::vector<ANPairType>::iterator iteJustDivAN;
        std::vector<RNPairType>::iterator iteJustDivRN;
        iteJustDivRN = find_if(justDividedCellsRN2.begin(), justDividedCellsRN2.end(), IsEqualLong(iteNewCoalesced->first));
        iteJustDivAN = justDividedCellsAN2.begin() + (iteJustDivRN - justDividedCellsRN2.begin());

        if (iteJustDivRN != justDividedCellsRN2.end() && iteJustDivAN != justDividedCellsAN2.end())
        {
            // we are in the case of a Red/Green cell, the newly coalesced region is also listed as just divided region
            // therefore, we need to cancelled the division, and decoalesced the region from this parasite tag

            RNPairType divisionRN = *iteJustDivRN;
            ANPairType divisionAN = *iteJustDivAN;
            if (iteNewCoalesced->first != iteJustDivRN->first)  // we arrange ourself to have the coalesced cell in first and the sister in second
            {
                divisionRN = SwitchPair(*iteJustDivRN);
                divisionAN = SwitchPair(*iteJustDivAN);
            }

            if (debugMode)
            {
                rglog << "coalesced division found: [" << iteJustDivRN->first << "] (" << iteJustDivAN->first << ") - [" << iteJustDivRN->second << "] (" << iteJustDivAN->second << ")" << std::endl;
                rglog << "\tthe coalesced one is : " << iteNewCoalesced->first << std::endl;
            }

            std::vector<CorrespondanceCouple>::iterator iteCorr;
            for (iteCorr = correspondance[currentFrame][iteNewCoalesced->first].begin(); iteCorr != correspondance[currentFrame][iteNewCoalesced->first].end(); ++iteCorr)
            {
                if (iteCorr->getAbsoluteTag().compare(divisionAN.first) == 0) // match
                {
                    //update reversetrack
                    if (debugMode)
                    {
                        rglog << "removing [" << iteCorr->getAbsoluteTag() << "] from cell #" << iteNewCoalesced->first << std::endl;
                    }
                    reverseTrack[currentFrame - 1][iteCorr->getAbsoluteTag()] = 0;            //remove first daughter
                    reverseTrack[currentFrame - 1][divisionAN.second] = 0;                    //remove second daughter
                    correspondance[currentFrame][divisionRN.first].erase(iteCorr);  // delete from correspondance

                    // cancel division
                    if (debugMode)
                    {
                        rglog << "changing [" << divisionAN.second << "] into ";
                    }
                    if (correspondance[currentFrame][divisionRN.second].size() > 1)
                    {
                        std::vector<CorrespondanceCouple>::iterator iteDivCoalesced;  // if the sister is also coalesced (extrem case)
                        for (iteDivCoalesced = correspondance[currentFrame][divisionRN.second].begin(); iteDivCoalesced != correspondance[currentFrame][divisionRN.second].end(); ++iteDivCoalesced)
                        {
                            if (iteDivCoalesced->getAbsoluteTag().compare(divisionAN.second) == 0) // match
                            {
                                iteDivCoalesced->undivided();
                            }
                        }
                    }
                    else
                    {
                        correspondance[currentFrame][divisionRN.second].front().undivided();
                    }

                    // update revertrack
                    std::string newAN = divisionAN.second.substr(0, divisionAN.second.size() - 2);
                    if (debugMode)
                    {
                        rglog << "[" << newAN << "] in cell #" << divisionRN.second << std::endl;
                    }
                    reverseTrack[currentFrame - 1][newAN] = divisionRN.second;

                    // remove from dividing cell
                    if (debugMode)
                    {
                        rglog << "updating dividing cells file" << std::endl;
                    }
                    bool found = false;
                    std::vector<unsigned long>::iterator iteDivCell;
                    for (iteDivCell = dividingCellsRN1.begin(); iteDivCell != dividingCellsRN1.end() && !found;)
                    {
                        std::string divCellAN = correspondance[currentFrame - 1][*iteDivCell].front().getAbsoluteTag();
                        if (divCellAN.compare(newAN) == 0)
                        {
                            if (debugMode)
                            {
                                rglog << "\twe delete the region #" << *iteDivCell << " from the DividingCellsRN list" << std::endl;
                            }
                            iteDivCell = dividingCellsRN1.erase(iteDivCell);
                            found = true;
                        }
                        else
                        {
                            iteDivCell++;
                        }
                    }

                    // remove from just divided cell
                    if (debugMode)
                    {
                        rglog << "updating just divided cells file" << std::endl;
                        rglog << "\twe delete " << iteJustDivRN->first << " , " << iteJustDivRN->second << " from the JDL RN" << std::endl;
                        rglog << "\twe delete " << iteJustDivAN->first << " , " << iteJustDivAN->second << " from the JDL AN" << std::endl;
                    }
                    justDividedCellsRN2.erase(iteJustDivRN);
                    justDividedCellsAN2.erase(iteJustDivAN);

                    // print the modification in the file
                    rgsolved << divisionRN.first << " " << divisionRN.second << std::endl;

                    // we clean coalesced list, raise the flag, and get out of the loop, we are done here
                    std::vector<std::string>::iterator iteToClean;
                    iteToClean = find(newlyCoalescedElements[divisionRN.first].begin(), newlyCoalescedElements[divisionRN.first].end(), divisionAN.first);
                    if (iteToClean != newlyCoalescedElements[divisionRN.first].end())
                    {
                        if (debugMode)
                        {
                            rglog << "\tWe remove the tag " << *iteToClean << " from the coalesced region #" << divisionRN.first << " in the newlycoalesced list" << std::endl;
                        }
                        newlyCoalescedElements[divisionRN.first].erase(iteToClean);
                    }
                    else
                    {
                        if (debugMode)
                        {
                            rglog << "\tthis should not happend, we could not find the tag in the newcoalescedlist to erase" << std::endl;
                        }
                    }

                    mainSolved = true;
                    break;
                }
            } // end loop for on correspondance
        } // end if found division
        else if (iteJustDivAN == justDividedCellsAN2.end() && iteJustDivRN == justDividedCellsRN2.end())
        {
            rglog << "\tRG - we did found SHIT in the just divided list couple" << std::endl;
        }
        else if (iteJustDivRN == justDividedCellsRN2.end())
        {
            rglog << "\tRG - we did not found the RN in the just divided list couple" << std::endl;
        }
        else if (iteJustDivAN == justDividedCellsAN2.end())
        {
            rglog << "\tRG - we did not found the AN in the just divided list couple" << std::endl;
        }


        if (correspondance[currentFrame][iteNewCoalesced->first].size() == 1 || iteNewCoalesced->second.empty())
        {
            if (debugMode)
            {
                rglog << "The coalesced region #" << iteNewCoalesced->first << " is solved, we remove it from the newlycoalesced list." << std::endl;
            }
            iteNewCoalesced = newlyCoalescedElements.erase(iteNewCoalesced);
        }
        else
        {
            if (debugMode)
            {
                rglog << "The coalesced region #" << iteNewCoalesced->first << " still contains some coalesced tag ... we move to the next one." << std::endl;
            }
            ++iteNewCoalesced;
        }



        //if (mainSolved)
        //{
        //    bool corr = correspondance[currentFrame][iteNewCoalesced->first].empty();
        //    if (iteNewCoalesced->second.empty() || corr)
        //    {
        //        iteNewCoalesced = newlyCoalescedElements.erase(iteNewCoalesced);
        //    }
        //    else
        //    {
        //        ++iteNewCoalesced;
        //    }
        //}
        //else
        //{
        //    ++iteNewCoalesced;
        //}

        if (debugMode)
        {
            rglog << std::endl;
        }
    } // end loop for on newly coalesced cell

    rgsolved.close();
    if (debugMode)
    {
        rglog.close();
    }
}

/* Red-Green Extended correstion patch
* @param newlyCoalescedElements the coalesced cell in the current frame
* @param correspondance the correpondance map
* @param neighbourMap the neighbor map for each cell
* @param borderCells2 the cell border list
* @param reverseTrack the reverse correspondance track
* @param dividingCellsRN1 the dividing cells (mother RN)
* @param justDividedCellsRN2 the just divided cells (daughter RN)
* @param justDividedCellsAN2 the just divided cells (daughter AN)
* @param currentFrame the frame index
* @param firstImage the first image index
* @param outputPath the output path
* @param debugMode the debug mode boolean
*/
void
RedGreenExtendedCorrectionPatch(std::map<unsigned long, std::vector<std::string> > &newlyCoalescedElements, std::map<int, std::map<int, std::vector<CorrespondanceCouple> > > &correspondance, std::map<unsigned long, std::vector<unsigned long> > &neighbourMap, std::vector<unsigned long> &borderCells2, std::map<std::string, unsigned long> *reverseTrack, std::vector<unsigned long> &dividingCellsRN1, std::vector<std::pair<unsigned long, unsigned long> > &justDividedCellsRN2, std::vector<std::pair<std::string, std::string> > &justDividedCellsAN2, unsigned int currentFrame, unsigned int firstImage, std::string outputPath, bool debugMode)
{
    // list of solved cases
    std::ofstream rgesolved((outputPath + "/Solved_RGe_" + intToString(firstImage + currentFrame - 1) + ".txt").c_str());
    std::ofstream rgelog;
    if (debugMode)
    {
        rgelog.open((outputPath + "/debugResolved_RGe" + intToString(firstImage + currentFrame - 1) + ".txt").c_str());
    }

    // for each coalesced cell left
    std::map<unsigned long, std::vector<std::string> >::iterator iteNewCoalesced;
    for (iteNewCoalesced = newlyCoalescedElements.begin(); iteNewCoalesced != newlyCoalescedElements.end(); /* ++iteNewCoalesced */)
    {         
        bool mainSolved = false;

        if (debugMode)
        {
            rgelog << "/****************************************************************************/" << std::endl;
            rgelog << "trying to solve coalescence of " << iteNewCoalesced->first << " (" << iteNewCoalesced->second.front();
            if (iteNewCoalesced->second.size() > 1)
            {
                for (std::vector<std::string>::iterator iteCoalTag = iteNewCoalesced->second.begin() + 1; iteCoalTag != iteNewCoalesced->second.end(); ++iteCoalTag)
                {
                    rgelog << " | " << *iteCoalTag;
                }
            }
            rgelog << ")" << std::endl;
        }

        // look for cell event in the 1-neighbourhood of the currentNewCell
        std::vector<unsigned long> newGreenNeighbor;
        std::vector<unsigned long>::iterator iteCellEvent;
        for (iteCellEvent = neighbourMap[iteNewCoalesced->first].begin(); iteCellEvent != neighbourMap[iteNewCoalesced->first].end(); ++iteCellEvent)
        {
            bool notBorderCell = find(borderCells2.begin(), borderCells2.end(), *iteCellEvent) == borderCells2.end();
            if (notBorderCell)
            {
                bool redFound(false), greenFound(false);
                std::pair<unsigned long, std::string> currentNeighbor = std::make_pair(*iteCellEvent, correspondance[currentFrame][*iteCellEvent].front().getAbsoluteTag());
                if (debugMode)
                {
                    rgelog << "\tthe neighbor cell " << currentNeighbor.first << "(" << currentNeighbor.second << ") -> ";
                }

                if (IsNewlyDividedRegion(justDividedCellsRN2, currentNeighbor.first))  // is the neighbor a newly divided cell ?
                {
                    newGreenNeighbor.push_back(currentNeighbor.first);
                    greenFound = true;
                    if (debugMode)
                    {
                        rgelog << "is newly divided";
                    }
                }
                if (IsCoalesced(newlyCoalescedElements, currentNeighbor.first)) // is the neighbor a newly coalesed cell ?
                {
                    //newRedNeighbor.push_back(currentNeighbor.first);
                    redFound = true;
                    if (debugMode)
                    {
                        if (greenFound)
                        {
                            rgelog << " && ";
                        }
                        rgelog << "is coalesced";
                    }
                }
                if (!redFound && !greenFound)
                {
                    if (debugMode)
                    {
                        rgelog << "is nothing";
                    }
                }
                if (debugMode)
                {
                    rgelog << std::endl;
                }
            } // end if not a border cell
        } // end loop for 1-neighbourhood event

        // 2 cases
        // 1 - it is a first layer cell -> we do nothing
        // 2 - it has a just divided cell next to her -> we solve it

        bool firstLayerFlag = IsFirstLayerCell(neighbourMap, borderCells2, iteNewCoalesced->first);
        if (firstLayerFlag)
        {
            if (debugMode)
            {
                rgelog << "It's a first layer cell, we do nothing. Bye bye !" << std::endl;
            }
        } // end case 1
        else if (!newGreenNeighbor.empty())
        {
            if (debugMode)
            {
                rgelog << newGreenNeighbor.size() << " divisions were detected in the direct neighborhood, we investigate ..." << std::endl;
            }

            std::vector<double>                          distances;
            std::vector<ANPairType>                      pairingCell;
            std::vector<RNPairType>                      pairingRegion;
            std::vector<std::pair< RNPairType, ANPairType > > pairingDivided;

            std::vector<unsigned long>::iterator iteGreenNeighbor;
            for (iteGreenNeighbor = newGreenNeighbor.begin(); iteGreenNeighbor != newGreenNeighbor.end(); ++iteGreenNeighbor)  // for each just divided in 1-neighborhood
            {
                std::pair<unsigned long, std::string> currentDividedCell = std::make_pair(*iteGreenNeighbor, correspondance[currentFrame][*iteGreenNeighbor].front().getAbsoluteTag());
                if (debugMode)
                {
                    rgelog << "\ttrying to solve the divided cell neighbor " << currentDividedCell.first << " (" << currentDividedCell.second << ")" << std::endl;
                }

                std::vector<CorrespondanceCouple>::iterator iteCorr;
                for (iteCorr = correspondance[currentFrame][iteNewCoalesced->first].begin();
                    iteCorr != correspondance[currentFrame][iteNewCoalesced->first].end(); iteCorr++)  // for each tag in the coalesced
                {
                    double redX = iteCorr->GetCentroid(0);  // coalesced centroid X
                    double redY = iteCorr->GetCentroid(1);  // coalesced centroid Y
                    double greenX = correspondance[currentFrame][currentDividedCell.first].front().GetCentroid(0);  // div cell centroid X
                    double greenY = correspondance[currentFrame][currentDividedCell.first].front().GetCentroid(1);  // div cell centroid Y

                    pairingRegion.push_back(std::make_pair(iteNewCoalesced->first, currentDividedCell.first));
                    pairingCell.push_back(std::make_pair(iteCorr->getAbsoluteTag(), currentDividedCell.second));
                    distances.push_back((greenX - redX) * (greenX - redX) + (greenY - redY) * (greenY - redY));

                    // pairing div (region,cell)
                    std::vector<ANPairType>::iterator iteSearchDivAN = find_if(justDividedCellsAN2.begin(), justDividedCellsAN2.end(), IsEqualStr(pairingCell.back().second));
                    std::vector<RNPairType>::iterator iteSearchDivRN = justDividedCellsRN2.begin() + (iteSearchDivAN - justDividedCellsAN2.begin());

                    RNPairType divRegion;
                    ANPairType divCell;
                    bool divisionFound = false;
                    if (iteSearchDivAN != justDividedCellsAN2.end() && iteSearchDivRN != justDividedCellsRN2.end())
                    {
                        divRegion = *iteSearchDivRN;
                        divCell   = *iteSearchDivAN;
                        divisionFound = true;
                    }
                    else
                    {
                        divRegion = std::make_pair(0, 0);
                        divCell = std::make_pair("0", "0");
                    }
                    pairingDivided.push_back(std::make_pair(divRegion, divCell));

                    if (debugMode)
                    {
                        rgelog << "\tdistance (squared) between newly divided cell: " << currentDividedCell.first << " (" << currentDividedCell.second << ") ";
                        rgelog << "... and new coalesced cell: " << iteNewCoalesced->first << " (" << iteCorr->getAbsoluteTag() << ") = " << distances.back() << std::endl;
                        if (divisionFound)
                        {
                            rgelog << "\t\tdetected division tag : " << pairingDivided.back().first.first << "," << pairingDivided.back().first.second
                                << " (" << pairingDivided.back().second.first << "," << pairingDivided.back().second.second << ")." << std::endl;
                        }
                        else
                        {
                            rgelog << "\t\twe did not detected a division ... this is ackward ..." << std::endl;
                        }
                    }
                } // end for loop for each tag in coalesced
            }  // end for loop for each divided suspect

            // get Minimum
            unsigned int  minindix = MinimumIndex(distances);

            // file print
            if (debugMode)
            {
                rgelog << "coalesced cell " << pairingRegion[minindix].first << "(" << pairingCell[minindix].first << ") found near the newly divided cells : ";
                rgelog << pairingRegion[minindix].second << " (" << pairingCell[minindix].second << ") " << std::endl;
                rgelog << "we kill the division [" << pairingDivided[minindix].first.first << "," << pairingDivided[minindix].first.second << "] -> [" << pairingDivided[minindix].second.first << "," << pairingDivided[minindix].second.second << "]" << std::endl;
            }
            // update correspondance
            std::vector<CorrespondanceCouple>::iterator iteCorr;
            for (iteCorr = correspondance[currentFrame][pairingRegion[minindix].first].begin(); iteCorr != correspondance[currentFrame][pairingRegion[minindix].first].end(); iteCorr++)
            {
                if (iteCorr->getAbsoluteTag().compare(pairingCell[minindix].first) == 0)
                {
                    // prepare for the handling the division
                    std::pair<unsigned long, std::string> sister = std::make_pair(0, "0");
                    std::vector<ANPairType>::iterator iteSearchDivAN;
                    std::vector<RNPairType>::iterator iteSearchDivRN;
                    iteSearchDivAN = find_if(justDividedCellsAN2.begin(), justDividedCellsAN2.end(), IsEqualStr(pairingCell[minindix].second));
                    iteSearchDivRN = find_if(justDividedCellsRN2.begin(), justDividedCellsRN2.end(), IsEqualLong(pairingRegion[minindix].second));
                    if (pairingCell[minindix].second.compare(iteSearchDivAN->first) == 0)
                    {
                        sister.second = iteSearchDivAN->second;
                        sister.first = iteSearchDivRN->second;
                    }
                    else
                    {
                        sister.second = iteSearchDivAN->first;
                        sister.first = iteSearchDivRN->first;
                    }

                    if (sister.first != pairingRegion[minindix].first)
                    {
                        // We take care of the coalesced tag 
                        if (debugMode)
                        {
                            rgelog << "\twe move the cell " << pairingCell[minindix].first << " in the region #" << pairingRegion[minindix].second << std::endl;
                        }
                        reverseTrack[currentFrame - 1][pairingCell[minindix].first] = pairingRegion[minindix].second;  // update coalesced reverse track
                        if (debugMode)
                        {
                            rgelog << "\twe clean the region #" << pairingRegion[minindix].second << " and associate the cell " << iteCorr->getAbsoluteTag() << std::endl;
                        }
                        correspondance[currentFrame][pairingRegion[minindix].second].clear();             // clear first daughter
                        correspondance[currentFrame][pairingRegion[minindix].second].push_back(*iteCorr); // add the newly coalesced cell in the new cell
                        if (debugMode)
                        {
                            rgelog << "\twe remove the cell " << iteCorr->getAbsoluteTag() << " from the coalesced region #" << pairingRegion[minindix].first << std::endl;
                        }
                        correspondance[currentFrame][pairingRegion[minindix].first].erase(iteCorr); // delete from correspondance


                        // We take care of the division
                        if (debugMode)
                        {
                            rgelog << "\twe delete the division region (" << iteSearchDivRN->first << "," << iteSearchDivRN->second;
                            rgelog << ") also know as division cell (" << iteSearchDivAN->first << "," << iteSearchDivAN->second << ")" << std::endl;
                        }
                        justDividedCellsRN2.erase(iteSearchDivRN);
                        justDividedCellsAN2.erase(iteSearchDivAN);
                        if (debugMode)
                        {
                            rgelog << "\tthe region #" << sister.first << " did not divide." << std::endl;
                        }
                        if (correspondance[currentFrame][sister.first].size() > 1)
                        {
                            std::vector<CorrespondanceCouple>::iterator iteDivCoalesced;  // if the sister is also coalesced (extrem case)
                            for (iteDivCoalesced = correspondance[currentFrame][sister.first].begin(); iteDivCoalesced != correspondance[currentFrame][sister.first].end(); ++iteDivCoalesced)
                            {
                                if (iteDivCoalesced->getAbsoluteTag().compare(sister.second) == 0) // match
                                {
                                    iteDivCoalesced->undivided();
                                }
                            }
                        }
                        else
                        {
                            correspondance[currentFrame][sister.first].front().undivided();
                        }

                        if (debugMode)
                        {
                            rgelog << "\tupdate reverse track for both divided cell" << std::endl;
                        }
                        std::string newName = sister.second.substr(0, sister.second.size() - 2);
                        reverseTrack[currentFrame - 1][pairingCell[minindix].second] = 0;
                        reverseTrack[currentFrame - 1][sister.second] = 0;
                        reverseTrack[currentFrame - 1][newName] = sister.first;

                        if (debugMode)
                        {
                            rgelog << "\twe remove the region from the mother division list" << std::endl;
                        }
                        std::vector<unsigned long>::iterator iteDivRN1Search1, iteDivRN1Search2;
                        iteDivRN1Search1 = find(dividingCellsRN1.begin(), dividingCellsRN1.end(), pairingRegion[minindix].second);
                        iteDivRN1Search2 = find(dividingCellsRN1.begin(), dividingCellsRN1.end(), sister.first);
                        if (iteDivRN1Search1 != dividingCellsRN1.end())
                        {
                            dividingCellsRN1.erase(iteDivRN1Search1);
                        }
                        else if (iteDivRN1Search2 != dividingCellsRN1.end())
                        {
                            dividingCellsRN1.erase(iteDivRN1Search2);
                        }

                        if (debugMode)
                        {
                            rgelog << "coalesced cell " << pairingRegion[minindix].first << "(" << pairingCell[minindix].first << ") moved in ";
                            rgelog << pairingRegion[minindix].second << " (" << pairingCell[minindix].second << ") and we erase the division ";
                            rgelog << "in the sister cell " << sister.first << "(" << sister.second << ")" << std::endl;
                        }

                        // print the modification in the file
                        rgesolved << pairingRegion[minindix].first << " " << pairingRegion[minindix].second << " " << sister.first << std::endl;

                        // we clean coalesced list, raise the flag, and get out of the loop, we are done here
                        std::vector<std::string>::iterator iteToClean;
                        iteToClean = find(iteNewCoalesced->second.begin(), iteNewCoalesced->second.end(), pairingCell[minindix].first);
                        if (iteToClean != iteNewCoalesced->second.end())
                        {
                            iteNewCoalesced->second.erase(iteToClean);
                        }
                        else
                        {
                            if (debugMode)
                            {
                                rgelog << "\tWe just solve an old coalesced tag ..." << std::endl;
                            }
                        }

                        mainSolved = true;
                        break;
                    }
                    else
                    {
                        if (debugMode)
                        {
                            rgelog << "The division sister is the coalesced cell that we try to solve ... this should have been handle in the RG patch ... we do nothing." << std::endl;
                        }
                        break;
                    }
                }
            } // end loop for coalesced tag
        } // end case 2
        else
        {
            if (debugMode)
            {
                rgelog << "No division detected in the neighborhood, bye bye !" << std::endl;
            }
        }

        if (mainSolved)
        {
            if (debugMode)
            {
                rgelog << "Is the region #" << iteNewCoalesced->first << " still coalesced ? " << std::endl;
            }
            if (correspondance[currentFrame][iteNewCoalesced->first].size() == 1)
            {
                if (debugMode)
                {
                    rgelog << "\tOnly 1 tag left in the region, it not a coalesced anymore ..." << std::endl;
                }
                iteNewCoalesced = newlyCoalescedElements.erase(iteNewCoalesced);
            }
            else if (iteNewCoalesced->second.empty())
            {
                if (debugMode)
                {
                    rgelog << "\tOnly old tag left in the region, we stop there ..." << std::endl;
                }
                iteNewCoalesced = newlyCoalescedElements.erase(iteNewCoalesced);
            }
            else
            {
                if (debugMode)
                {
                    rgelog << "\tIt still contain at least 2 tag (and one of them is newlyCoalesced)" << std::endl;
                }
                ++iteNewCoalesced;
            }
        }
        else
        {
            ++iteNewCoalesced;
        }
        if (debugMode)
        {
            rgelog << std::endl;
        }
    }
}

/* Correspondance correction patch
* @param newlyCoalescedElements the coalesced cell in the current frame
* @param correspondance the correpondance map
* @param reverseTrack the reverse correspondance track
* @param borderCells2 the cell border list
* @param newCellsRN_current the list of new cells (RN)
* @param newCellsRN_current the list of new cells (AN)
* @param centroidsArray the list of centroids for each cell
* @param numberOfCells1 the number of cell in the current frame
* @param numberOfCells2 the number of cell in the previous frame
* @param currentFrame the frame index
* @param firstImage the first image index
* @param outputPath the output path
* @param debugMode the debug mode boolean
*/
void
CorrespondanceCorrectionPatch(std::map<unsigned long, std::vector<std::string> > *newlyCoalescedElements, unsigned long &max, std::map<int, std::map<int, std::vector<CorrespondanceCouple> > > &correspondance, std::map<std::string, unsigned long> *reverseTrack, std::vector<unsigned long> &borderCells2, std::vector<unsigned long> &newCellsRN_current, std::vector<unsigned long> &newCellsAN_current, std::pair<double, double> *centroidsArray, unsigned int numberOfCells1, unsigned int numberOfCells2, unsigned int currentFrame, unsigned int firstImage, std::string outputPath, bool debugMode)
{
    CorrespondanceCouple correspondanceFiller;
    std::vector<CorrespondanceCouple> currentCorr;
    // Determination of the maximum number of assigned cells in the frame 1

    for (unsigned long currentCell = 1; currentCell <= numberOfCells1; currentCell++)
    {
        std::vector<CorrespondanceCouple>::iterator iteCorrCouple;
        currentCorr = correspondance[currentFrame - 1][currentCell];
        for (iteCorrCouple = currentCorr.begin(); iteCorrCouple != currentCorr.end(); iteCorrCouple++)
        {
            if (iteCorrCouple->GetAbsoluteNumber() > max)
            {
                max = iteCorrCouple->GetAbsoluteNumber();
            }
        }
    }

    // Assignment of an absolute number for the cells of frame 2 which do not have already one and determine coalesced cells
    for (unsigned long currentCell = 1; currentCell <= numberOfCells2; currentCell++)
    {
        // FILLS UP coalesced_cells_RN
        if (correspondance[currentFrame][currentCell].size() > 1)
        {
            if (find(borderCells2.begin(), borderCells2.end(), currentCell) == borderCells2.end())
            {
                //bool notFound, notCoalesced;
                std::string absoluteTag;
                std::map<std::string, unsigned long>::iterator iteReverse;
                std::vector<CorrespondanceCouple>::iterator iteCorr;
                currentCorr = correspondance[currentFrame][currentCell];
                for (iteCorr = currentCorr.begin(); iteCorr != currentCorr.end(); iteCorr++)
                {
                    absoluteTag = iteCorr->getAbsoluteTag();                        // get absolute tag
                    iteReverse = reverseTrack[currentFrame - 2].find(absoluteTag);  // find corresponding cell in previous frame

                    if (iteReverse == reverseTrack[currentFrame - 2].end() ||
                        (iteReverse != reverseTrack[currentFrame - 2].end() && correspondance[currentFrame - 1][iteReverse->second].size() == 1))
                    {
                        // fill the newly coalesced map (2012-07-09)
                        std::map<unsigned long, std::vector<std::string> >::iterator iteNewCoalesced = newlyCoalescedElements->find(currentCell);
                        if (iteNewCoalesced == newlyCoalescedElements->end())
                        {
                            newlyCoalescedElements->insert(std::make_pair(currentCell, std::vector<std::string>(1, absoluteTag)));
                        }
                        else
                        {
                            iteNewCoalesced->second.push_back(absoluteTag);
                        }
                    }
                }
            }
        }
        else if (correspondance[currentFrame][currentCell].empty()) // fill the empty spaces with new cells
        {
            max++;
            correspondanceFiller.Delete();                                                // reset filler
            correspondanceFiller.SetAbsoluteNumber(max);                                  // set Absolute number
            correspondanceFiller.SetCentroid(0, centroidsArray[currentCell - 1].first);   // set centroid X
            correspondanceFiller.SetCentroid(1, centroidsArray[currentCell - 1].second);  // set cendtroid Y
            correspondance[currentFrame][currentCell].push_back(correspondanceFiller);    // fill correspondance
            newCellsRN_current.push_back(currentCell);                                    // store RN of cell not assigned during previous iteration
            newCellsAN_current.push_back(max);                                            // store AN of cell not assigned during previous iteration
        }
        // build reverseTrack map for current frame (2011-07-22)
        std::vector<CorrespondanceCouple>::iterator iteCorrCouple;
        currentCorr = correspondance[currentFrame][currentCell];
        for (iteCorrCouple = currentCorr.begin(); iteCorrCouple != currentCorr.end(); iteCorrCouple++)
        {
            reverseTrack[currentFrame - 1][iteCorrCouple->getAbsoluteTag()] = currentCell;
        }
    } // end for loop on cell in frame 2 //delete[] centroidsArray;  // deallocate memory for centroids array

    // print the newly coalesced cells list
    if (debugMode)
    {
        std::ofstream newCorrespondance((outputPath + "/debugNewlyCoalesced_BF_" + intToString(firstImage + currentFrame - 1) + ".txt").c_str());
        std::map<unsigned long, std::vector<std::string> >::iterator iteNewCoalesced;
        for (iteNewCoalesced = newlyCoalescedElements->begin(); iteNewCoalesced != newlyCoalescedElements->end(); iteNewCoalesced++)
        {
            newCorrespondance << iteNewCoalesced->first;
            std::vector<std::string>::iterator iteInner;
            for (iteInner = iteNewCoalesced->second.begin(); iteInner != iteNewCoalesced->second.end(); iteInner++)
            {
                newCorrespondance << " '" << *iteInner << "'";
            }
            newCorrespondance << std::endl;
        }
    }
}


// ************************************************************************************************************************************************************************************************* \\

/*** MAIN PROGRAM
* arguments (in the following order):
* (+) fullpath to the first segmented image
* (+) output directory (without the '/' at the end)
* (+) first image number
* (+) last image number
* (+) fullpath to the first PIV file (3.0)
* (+) correction options
* (+) other options (3.3)
* arguments (alternative call)
* (+) parameters file path
* the file should have the following structure (parameter order is not important)
*parameter *value
SPATH  segmentation first image path (MANDATORY)
OPATH  output path (MANDATORY)
FIRST  first frame number (MANDATORY)
LAST   last frame number (MANDATORY)
PPATH  PIV path (MANDATORY)
COPT   correction disabling option (default: 110)
OOPT   other option (default: 0)
MDR2   mother daughter ratio 2 (default: 1.0)
MDR3   mother daughter ratio 3 (default: 0.7)
DDR2   daughter daughter ratio 2 (default: 0.1)
DDR3   daughter daughter ratio 3 (default: 0.3)
ANISO  anisotropy threshold (default: 0.8)
APOP   apoptosis threashold (default: 0.3)
NAPO   number of frame for apoptosis (default: 3)
*/
int main(int argc, char *argv[])       // Now calling input arguments in the main function (2.5d):
{
    std::cout << "Tracking " << TRACKING_VERSION << " ::: compilation time: " << __DATE__ << " " << __TIME__ << std::endl;

    // for benchmark
    clock_t t_start, t_end;
    t_start = clock();

    // ---------- PARAMETERS AND VARIABLE INITIALISATION ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    std::string       path = "";     // fullpath to first segmented image
    std::string       outputPath = "";     // output directory
    unsigned int firstImage = 0;      // first frame
    unsigned int lastImage = 0;      // last frame
    std::string       uFirstPIVPath = "";     // fullpath to first->second image PIV
    std::string       correctionOption = "1110";  // 1 for apoptosis, 1 for red/blue, 0 for red/green
    std::string       otheroptions = "";     // otheroption
    bool         resumeMode = false;  // resume tracking
    bool         debugMode = true;   // debug prints (2012-06-25)

    // Tracking parameters (default values) -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    double       motherDaughterRatio2 = 1.0;
    double       motherDaughterRatio3 = 0.7;  // threshold value of the ratio of the volumes of the daugther cells on the volume of the mother cell.
    double       daughterDaughterRatio2 = 0.1;
    double       daughterDaughterRatio3 = 0.3;
    unsigned int maximumDivisionAllowed = 5;
    double       anisotropyThreshold = 0.8;
    double       apoptosisThreshold = 0.3;  // threshold for apoptosis detection
    double       apopSizePencentil = 33;   // percentil of the cell size distribution use for apoptosis size threshold
    unsigned int apoptosisCheckFrames = 3;    // number of frames checked for apoptosis detection
    unsigned int ExtendedCentroidRadius = 900;  // squared radius for centroid search in "no-overlap case"

    bool transposeFlag = false;
    itk::FixedArray<int, 2> order;
    order[0] = 0;
    order[1] = 1;



    // parameters provided in a file --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if (argc == 2)
    {
        std::cout << "Loading tracking parameters from file ... " << std::endl;
        std::ifstream pfile(argv[1]);
        if (pfile.is_open())
        {
            int out, parameterFail = 0;
            out = ReadParams(pfile, "SPATH", path);
            if (out == 0)
            {
                std::cerr << "Error 'Main' : Segmentation file path not found!" << std::endl;
                parameterFail++;
            }
            out = ReadParams(pfile, "OPATH", outputPath);
            if (out == 0)
            {
                std::cerr << "Error 'Main' : Output path not found!" << std::endl;
                parameterFail++;
            }
            out = ReadParams(pfile, "FIRST", firstImage);
            if (out == 0)
            {
                std::cerr << "Error 'Main' : First frame number not found!" << std::endl;
                parameterFail++;
            }
            out = ReadParams(pfile, "LAST", lastImage);
            if (out == 0)
            {
                std::cerr << "Error 'Main' : Last frame number not found!" << std::endl;
                parameterFail++;
            }
            out = ReadParams(pfile, "PPATH", uFirstPIVPath);
            if (out == 0)
            {
                std::cerr << "Error 'Main' : PIV path not found!" << std::endl;
                parameterFail++;
            }
            // optionnal parameters
            ReadParams(pfile, "COPT", correctionOption);
            ReadParams(pfile, "OOPT", otheroptions);
            ReadParams(pfile, "MDR2", motherDaughterRatio2);
            ReadParams(pfile, "MDR3", motherDaughterRatio3);
            ReadParams(pfile, "DDR2", daughterDaughterRatio2);
            ReadParams(pfile, "DDR3", daughterDaughterRatio3);
            ReadParams(pfile, "MDA", maximumDivisionAllowed);
            ReadParams(pfile, "ANISO", anisotropyThreshold);
            ReadParams(pfile, "APOP", apoptosisThreshold);
            ReadParams(pfile, "NAPO", apoptosisCheckFrames);
            ReadParams(pfile, "CAPO", apopSizePencentil);
            ReadParams(pfile, "CRAD", ExtendedCentroidRadius);
            // close the file
            pfile.close();
            // incomplete?
            if (parameterFail != 0)
            {
                std::cerr << "Error 'Main' : PARAMETERS FILE INCOMPLETE ..." << parameterFail << " parameters missing!" << std::endl;
                return EXIT_FAILURE;
            }
        }
        else
        {
            std::cerr << "Error 'Main' : Unable to open parameters file " << argv[1] << std::endl;
            return EXIT_FAILURE;
        }
    }
    else // TRACKING WITH DEFAULT PARAMETERS
    {
        path = argv[1];
        outputPath = argv[2];
        firstImage = atoi(argv[3]);
        lastImage = atoi(argv[4]);
        uFirstPIVPath = argv[5];
        // correction option
        if (argc > 6)
        {
            correctionOption = argv[6];
        }
        // option
        if (argc == 8)
        {
            otheroptions = argv[7];
        }
    }

    // setup OPTIONS ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if (correctionOption.size() < 3)
    {
        std::cerr << "Warning 'Main' : Invalid CorrectionOption ... Not enough digits (3 required, " << correctionOption.size() << " provided). Using default parameter (110) instead." << std::endl;
        correctionOption = "110";
    }
    if (otheroptions.compare("") != 0)
    {
        if (otheroptions.find('R') != std::string::npos)
        {
            resumeMode = true;
        }
        if (otheroptions.find('D') != std::string::npos)
        {
            debugMode = true;
        }
        if (otheroptions.find('T') != std::string::npos)
        {
            std::cout << "Tracking: Transpose mode activated!" << std::endl;
            transposeFlag = true;
            order[0] = 1;
            order[1] = 0;
            
        }
    }

    // Segmentation file parsing ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    // EXTRACT fileRootNAME, fileExtension & NDIGIT for segmentation files
    // slicing filename into [fileRootname]+[filenumber]+[fileExtension]
    size_t found = path.find_last_of('.');
    if (found == std::string::npos)
    {
        std::cerr << "Error 'Main' : Invalid segmentation file path ... extension not found!" << std::endl;
        return EXIT_FAILURE;
    }

    std::string segfileExtension = path.substr(found);      // file extension (includes the .)
    std::string segfileRoot = path.substr(0, found);   // fileRootname+filenumber
    unsigned int iTemp = segfileRoot.size() - 1;  // determine the number of digits
    unsigned int segNDigit = 0;
    while (isdigit(segfileRoot[iTemp--]))
    {
        segNDigit++;
    }
    segfileRoot = segfileRoot.substr(0, segfileRoot.size() - segNDigit);  // fileRootname

    // PIV file parsing ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    // EXTRACT fileRootNAME, fileExtension & NDIGIT for PIV files (2011-07-04)
    // PIV filename slicing
    found = uFirstPIVPath.find_last_of('.');
    // file extension
    std::string fileExtension = uFirstPIVPath.substr(found);
    std::string fileRoot = uFirstPIVPath.substr(0, found);
    found = fileRoot.find_last_of('_');
    // file number
    std::string nFileS = fileRoot.substr(found + 1);
    fileRoot = fileRoot.substr(0, found - 1);
    /* PIV filename: fileRoot+["u_"|"v_"]+number+fileExtension */
    unsigned int NDigitPIV = nFileS.size();  // number of digit

    // number of frames
    int numberOfFrames = lastImage - firstImage + 1;

    // read GRID SIZE direclty from PIV filename (2012-10-26)
    std::string junk, line = "";
    pathspliter(uFirstPIVPath, junk, line);  // split in folder and file
    // look for grid size: pivfilenameroot= PIV+_+[gridsize]+_[animal]+_
    found = line.find_first_of('_');
    line = line.substr(found + 1);
    found = line.find_first_of('_');
    line = line.substr(0, found);

    // print parameters list ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    // tracking parameters list (2012-05-03)
    std::ofstream pOutFile((outputPath + "/CppMT_parameters_" + intToString(firstImage) + "-" + intToString(lastImage) + ".txt").c_str());
    pOutFile << "Processing start time: " << __DATE__ << " " << __TIME__ << std::endl;
    pOutFile << "first frame segmentation: " << path << std::endl;
    pOutFile << "output folder: " << outputPath << std::endl;
    pOutFile << "first frame: " << firstImage << std::endl;
    pOutFile << "last frame: " << lastImage << std::endl;
    pOutFile << "first frame PIV: " << uFirstPIVPath << std::endl;
    if (!line.empty())
    {
        pOutFile << line << std::endl; //2012-05-11
    }
    else
    {
        pOutFile << "PIV Grid size: n/a" << std::endl;
    }
    pOutFile << "correction option: " << correctionOption << std::endl;
    pOutFile << "other options: " << otheroptions << std::endl;
    pOutFile << "Tracking parameters:" << std::endl;
    pOutFile << "\tDivision:" << std::endl;
    pOutFile << "mother/daughter volume ratio thresholds: " << motherDaughterRatio2 << " " << motherDaughterRatio3 << std::endl;
    pOutFile << "daughter/daughter volume ratio thresholds: " << daughterDaughterRatio2 << " " << daughterDaughterRatio3 << std::endl;
    pOutFile << "anisotropyThreshold threshold: " << anisotropyThreshold << std::endl;
    pOutFile << "\tApoptosis:" << std::endl;
    pOutFile << "apoptosis volume decrease threshold: " << apoptosisThreshold << std::endl;
    pOutFile << "apoptosis check frame: " << apoptosisCheckFrames << std::endl;
    pOutFile.close();

    // PIV grid parameters (2014-06-20 - 3.6)
    std::ofstream bOutFile((outputPath + "/PIV_Grid.txt").c_str());
    if (!line.empty())
    {
        bOutFile << line << std::endl;
    }
    else
    {
        bOutFile << "PIV Grid size: n/a" << std::endl;
    }
    bOutFile.close();



    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    // Declaration of the correspondance map, which has number of frame as a dimension, ie, which is not intern in the loop on frames
    std::map<int, std::map<int, std::vector<CorrespondanceCouple> > > correspondance;
    CorrespondanceCouple correspondanceFiller;

    // Declaration of the label area map and reverse correspondance map, also of number of frame as dimension and outside the main loop
    std::map<unsigned long, unsigned long> *labelsAreas = new std::map<unsigned long, unsigned long>[numberOfFrames];  // (~APOP) datastructure to store the labels area (2011-07-12)
    std::map<std::string, unsigned long> *reverseTrack = new std::map<std::string, unsigned long>[numberOfFrames];         // reverse correspondance map (2011-07-20)

    // Declaration of the border cell lists. Outside of the loop to keep in memory the lists of the previous iteration, ie, list2 into list1
    std::vector<unsigned long> borderCells1;
    std::vector<unsigned long> borderCells2;

    // number of divisions encountered (2012-05-03)
    unsigned short nbDivision = 0;

    // ---------- TRACKING MAIN ITERATION INITIALISATION ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    // First Image
    // The first image is treated as a separate case initialisation to avoid if statement into the main loop
    // So the loop on frame will start at frame 2
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Definition of the neighbor pixels on which the iteration will be done
    ConstNeighborhoodIteratorType::OffsetType offset1 = { { -1, -1 } };
    ConstNeighborhoodIteratorType::OffsetType offset2 = { { -1, 0 } };
    ConstNeighborhoodIteratorType::OffsetType offset3 = { { -1, 1 } };
    ConstNeighborhoodIteratorType::OffsetType offset4 = { { 0, -1 } };
    ConstNeighborhoodIteratorType::OffsetType offset5 = { { 0, 1 } };
    ConstNeighborhoodIteratorType::OffsetType offset6 = { { 1, -1 } };
    ConstNeighborhoodIteratorType::OffsetType offset7 = { { 1, 0 } };
    ConstNeighborhoodIteratorType::OffsetType offset8 = { { 1, 1 } };
    ConstNeighborhoodIteratorType::RadiusType radius;
    radius.Fill(1);

    // read the first image
    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName(path.c_str());
    TransposeImageFilterType::Pointer transpose = TransposeImageFilterType::New();
    transpose->SetInput(reader->GetOutput());
    transpose->SetOrder(order);
    // calculate connex component label
    LabelFilterType::Pointer labelfilter = LabelFilterType::New();
    labelfilter->SetInput(transpose->GetOutput());
    // calculate geometrical feature for each labeled object
    GeometryFilterType::Pointer geometryfilter = GeometryFilterType::New();
    geometryfilter->SetInput(labelfilter->GetOutput());
    try
    {
        geometryfilter->Update();  // Update itk pipeline
    }
    catch (itk::ExceptionObject &excp)
    {
        std::cerr << "Error itk : Unable to read and extract geometrical information from image." << std::endl;
        std::cerr << excp << std::endl;
        return EXIT_FAILURE;
    }
    // get image size
    ITKImageType::SizeType imageSize = transpose->GetOutput()->GetLargestPossibleRegion().GetSize();

    // Resume tracking (2012-05-18)
    if (resumeMode)
    {
        std::cout << "Resuming from image # " << firstImage << " ... " << std::endl;

        // read correspondance file
        std::string line;
        std::ifstream corr((outputPath + "/correspondance#_" + intToString(firstImage) + ".txt").c_str());
        if (corr.is_open())
        {
            unsigned long cellRN;
            std::string tag;
            double cellcX;
            double cellcY;
            short jd;

            // read the file contents
            while (corr.good())
            {
                getline(corr, line); // read the line

                if (line.empty())
                {
                    break;
                }

                // line in the std::stringstream
                std::stringstream ss(line);
                // format is : %d %s %f %f %d
                ss >> cellRN >> tag >> cellcX >> cellcY >> jd;

                // split the tag into absolute number and division tag
                size_t found = tag.find_first_of('_');
                unsigned long an;
                if (found == std::string::npos) // no division tag
                {
                    an = atoi(tag.c_str());
                    tag = "";
                }
                else
                {
                    an = atoi(tag.substr(0, found).c_str());
                    tag = tag.substr(found);
                    replace(tag.begin(), tag.end(), '_', ' '); // re-format division tag
                }

                // add the element in the correspondance map
                correspondanceFiller.Delete();
                correspondanceFiller.SetAbsoluteNumber(an);
                if (!tag.empty())
                {
                    correspondanceFiller.SetTag(tag);
                }
                correspondanceFiller.SetCentroid(0, cellcX);
                correspondanceFiller.SetCentroid(1, cellcY);
                if (jd)
                {
                    correspondanceFiller.setJustDivided();
                }
                correspondance[1][cellRN].push_back(correspondanceFiller);
                reverseTrack[0][correspondanceFiller.getAbsoluteTag()] = cellRN;  // fill reverse track
                labelsAreas[0][cellRN] = geometryfilter->GetVolume(cellRN);       // fill labelsAreas
            }
            corr.close();
        }
        else
        {
            std::cerr << "Error 'Main' : Unable to open file " << outputPath + "/correspondance#_" + intToString(firstImage) + ".txt" << std::endl;
            return EXIT_FAILURE;
        }
        // read bordercells file
        ReadBorderCellsFile((outputPath + "/border_cells_RN_" + intToString(firstImage) + ".txt").c_str(), borderCells1);
    }
    else // classic run
    {
        std::cout << "Initializing image # " << firstImage << " ... " << std::endl; // 2.5c

        //Creation of the correspondance table for the first frame and txt for cell areas (~APOP 2011-07-12)
        std::ofstream corr((outputPath + "/correspondence_" + intToString(firstImage) + ".txt").c_str());
        std::ofstream corrbackup((outputPath + "/correspondance#_" + intToString(firstImage) + ".txt").c_str()); // 2012-05-09
        for (unsigned long currentCell = 1; currentCell < geometryfilter->GetNumberOfLabels(); currentCell++)
        {
            // reset correspondance
            correspondanceFiller.Delete();
            correspondanceFiller.SetAbsoluteNumber(currentCell);

            // incoherent value check
            if (geometryfilter->GetCentroid(currentCell)[0] < 0 || geometryfilter->GetCentroid(currentCell)[1] < 0)  // negative centroid calculated by itklabelgeometryfilter, that should not happen
            {
                std::cerr << "WARNING 'Main' : negative centroid #" << currentCell << " (" << geometryfilter->GetCentroid(currentCell)[0] << ";" << geometryfilter->GetCentroid(currentCell)[1] << ") - detected from ITK" << std::endl;
                if (geometryfilter->GetCentroid(currentCell)[0] < 0)
                {
                    geometryfilter->GetCentroid(currentCell)[0] = -geometryfilter->GetCentroid(currentCell)[0];
                }
                if (geometryfilter->GetCentroid(currentCell)[1] < 0)
                {
                    geometryfilter->GetCentroid(currentCell)[1] = -geometryfilter->GetCentroid(currentCell)[1];
                }
            }

            // save geometry values
            correspondanceFiller.SetCentroid(0, geometryfilter->GetCentroid(currentCell)[0]);
            correspondanceFiller.SetCentroid(1, geometryfilter->GetCentroid(currentCell)[1]);
            correspondance[1][currentCell].push_back(correspondanceFiller);
            labelsAreas[0][currentCell] = geometryfilter->GetVolume(currentCell);   // number of pixels for the label i (~APOP) (2011-07-12)
            corr << currentCell << " " << currentCell << std::endl;                      // print correspondance (2012-05-04)
            corrbackup << currentCell << ' ' << correspondanceFiller.getAbsoluteTag_S() << ' ' << correspondanceFiller.GetCentroid(0) << ' ' << correspondanceFiller.GetCentroid(1) << ' ' << correspondanceFiller.isJustDivided() << std::endl;  // 2012-05-16
            reverseTrack[0][correspondanceFiller.getAbsoluteTag()] = currentCell;                 // fill reverseTrack (2012-05-23)

        }  // end for loop on cell
        corr.close();
        corrbackup.close();

        // Coalesced cells RN (empty set in 1st frame):
        std::fstream coalescedCellsRN((outputPath + "/coalesced_cells_RN_" + intToString(firstImage) + ".txt").c_str(), std::ios::out);  // moved inside loop + specify the frame number Bo
        coalescedCellsRN << -1 << std::endl;
        coalescedCellsRN.close();

        // Just divided cells next RN:
        std::fstream justDividedCells((outputPath + "/just_divided_cells_RN_" + intToString(firstImage) + ".txt").c_str(), std::ios::out); // **** n+1 **** moved inside loop + specify the frame number Bo
        justDividedCells << -1 << std::endl;
        justDividedCells.close();

        // New cells AN (empty set in 1st frame)
        std::fstream newCellsAN((outputPath + "/new_cells_AN_" + intToString(firstImage) + ".txt").c_str(), std::ios::out);              // moved inside loop + specify the frame number Bo
        newCellsAN << -1 << std::endl;
        newCellsAN.close();

        // New cells RN (empty set in 1st frame)
        std::fstream newCellsRN((outputPath + "/new_cells_RN_" + intToString(firstImage) + ".txt").c_str(), std::ios::out);              // moved inside loop + specify the frame number Bo
        newCellsRN << -1 << std::endl;
        newCellsRN.close();

        // fill the neighbour map (2012-02-21)
        std::map<unsigned long, std::vector<unsigned long> > neighbourMap;
        FillNeighboursList(labelfilter->GetOutput(), &neighbourMap, outputPath + "/skeletonPixelsList_" + intToString(firstImage) + ".txt", borderCells1);
        PrintNeighboursMap(outputPath + "/neighbours_RN_" + intToString(firstImage) + ".txt", neighbourMap);            // print the neighbours map
        PrintBordercells((outputPath + "/border_cells_RN_" + intToString(firstImage) + ".txt").c_str(), borderCells1);  // print the bordercells
    }

    // ---------- TRACKING MAIN ITERATION ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    for (int currentFrame = 2; currentFrame <= numberOfFrames; currentFrame++)
    {
        std::cout << "Processing image # " << firstImage + currentFrame - 1 << " ... " << std::endl;

        // new cells that just appeared in the current frame
        std::vector<unsigned long>                       newCellsRN_current;   // new cell relative number
        std::vector<unsigned long>                       newCellsAN_current;   // new cell absolute number
        std::vector<unsigned long>                       dividingCellsRN1;     // cells that will divided in the next frame
        std::vector<std::pair<unsigned long, unsigned long> > justDividedCellsRN2;  // pair of just divided cells
        std::vector<std::pair<std::string, std::string> > justDividedCellsAN2;  // pair of just divided cells
        std::vector<unsigned long>                       coalescedCellsRN2;    // coalesced cells

        // Declaration of the readers, the images and the filters. 1 will be related to image n-1 and 2 will be related to image n
        ReaderType::Pointer         segmentedImageReader1 = ReaderType::New();
        ReaderType::Pointer         segmentedImageReader2 = ReaderType::New();
        ReaderType::Pointer         pivReader1 = ReaderType::New();
        ReaderType::Pointer         pivReader2 = ReaderType::New();
        TransposeImageFilterType::Pointer transposeImage1 = TransposeImageFilterType::New();
        TransposeImageFilterType::Pointer transposeImage2 = TransposeImageFilterType::New();
        TransposeImageFilterType::Pointer transposePiv1 = TransposeImageFilterType::New();
        TransposeImageFilterType::Pointer transposePiv2 = TransposeImageFilterType::New();
        LabelFilterType::Pointer    labelFilter1 = LabelFilterType::New();
        LabelFilterType::Pointer    labelFilter2 = LabelFilterType::New();
        GeometryFilterType::Pointer geometryFilter1 = GeometryFilterType::New();
        GeometryFilterType::Pointer geometryFilter2 = GeometryFilterType::New();
        ITKImageType::Pointer       image1;
        ITKImageType::Pointer       image2;

        // read first and second image and extract label/geometry using itk filter ----------------------------------------------------------------------------------------------------------------------------------------------
        // current frame segmentation
        std::string readerPath2 = segfileRoot + numberToString(firstImage + currentFrame - 1, segNDigit) + segfileExtension;
        segmentedImageReader2->SetFileName(readerPath2.c_str());
        transposeImage2->SetInput(segmentedImageReader2->GetOutput());
        transposeImage2->SetOrder(order);
        labelFilter2->SetInput(transposeImage2->GetOutput());
        geometryFilter2->SetInput(labelFilter2->GetOutput());
        try
        {
            geometryFilter2->Update();  // Update itk pipeline
        }
        catch (itk::ExceptionObject &excp)
        {
            std::cerr << "Error itk : Unable to read and extract label information from image 2." << std::endl;
            std::cerr << excp << std::endl;
            return EXIT_FAILURE;
        }

        // previous frame segmentation
        std::string readerPath1 = segfileRoot + numberToString(firstImage + currentFrame - 2, segNDigit) + segfileExtension;
        segmentedImageReader1->SetFileName(readerPath1.c_str());
        transposeImage1->SetInput(segmentedImageReader1->GetOutput());
        transposeImage1->SetOrder(order);
        labelFilter1->SetInput(transposeImage1->GetOutput());
        geometryFilter1->SetInput(labelFilter1->GetOutput());
        try
        {
            geometryFilter1->Update();  // Update itk pipeline
        }
        catch (itk::ExceptionObject &excp)
        {
            std::cerr << "Error itk : Unable to read and extract label information from image 1." << std::endl;
            std::cerr << excp << std::endl;
            return EXIT_FAILURE;
        }

        // set images
        image1 = labelFilter1->GetOutput();
        image2 = labelFilter2->GetOutput();

        // Iterators on image1 and image2
        IteratorType iterator1(image1, image1->GetLargestPossibleRegion());
        IteratorType iterator2(image2, image2->GetLargestPossibleRegion());

        // update previous frame border cells
        if (!borderCells2.empty())
        {
            borderCells1 = borderCells2;
            borderCells2.clear();
        }

        // fill the neighbour map & the border cells ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        std::map<unsigned long, std::vector<unsigned long> > neighbourMap;
        FillNeighboursList(image2, &neighbourMap, outputPath + "/skeletonPixelsList_" + intToString(firstImage + currentFrame - 1) + ".txt", borderCells2);
        PrintNeighboursMap(outputPath + "/neighbours_RN_" + intToString(firstImage + currentFrame - 1) + ".txt", neighbourMap);            // print the neighbours map
        PrintBordercells((outputPath + "/border_cells_RN_" + intToString(firstImage + currentFrame - 1) + ".txt").c_str(), borderCells2);  // print the bordercells


        // Load PIV files (2012-10-26) ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if (debugMode)
        {
            std::cout << "\tLoading PIV ... " << std::endl;
        }

        // load U file
        pivReader1->SetFileName((fileRoot + "u_" + numberToString(firstImage + currentFrame - 2, NDigitPIV) + fileExtension).c_str());
        transposePiv1->SetInput(pivReader1->GetOutput());
        transposePiv1->SetOrder(order);
        try
        {
            transposePiv1->Update();
        }
        catch (itk::ExceptionObject &excp)
        {
            std::cerr << "Error itk : Unable to read PIV file U." << std::endl;
            std::cerr << excp << std::endl;
            return EXIT_FAILURE;
        }

        // load V file
        pivReader2->SetFileName((fileRoot + "v_" + numberToString(firstImage + currentFrame - 2, NDigitPIV) + fileExtension).c_str());
        transposePiv2->SetInput(pivReader2->GetOutput());
        transposePiv2->SetOrder(order);
        try
        {
            transposePiv2->Update();
        }
        catch (itk::ExceptionObject &excp)
        {
            std::cerr << "Error itk : Unable to read PIV file V." << std::endl;
            std::cerr << excp << std::endl;
            return EXIT_FAILURE;
        }

        ITKImageType::Pointer pivVImage;
        ITKImageType::Pointer pivUImage;
        if (transposeFlag)
        {
            pivVImage = transposePiv1->GetOutput();
            pivUImage = transposePiv2->GetOutput();
        }
        else
        {
            pivUImage = transposePiv1->GetOutput();
            pivVImage = transposePiv2->GetOutput();
        }

        // PIV iterator
        IteratorType iteratorUPIV(pivUImage, pivUImage->GetLargestPossibleRegion());  // PIV U iterator
        IteratorType iteratorVPIV(pivVImage, pivVImage->GetLargestPossibleRegion());  // PIV V iterator

        // get number of cells for both frames ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        unsigned int numberOfCells1 = geometryFilter1->GetNumberOfLabels() - 1;
        unsigned int numberOfCells2 = geometryFilter2->GetNumberOfLabels() - 1;

        // fill labels area -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        for (unsigned long iLabels = 1; iLabels <= numberOfCells2; iLabels++)
        {
            labelsAreas[currentFrame - 1][iLabels] = geometryFilter2->GetVolume(iLabels);
        }

        // ---------- IMAGE - IMAGE CELL COMPARISON -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        // this part creates the map overlap which gives for each cell of the image1
        //  - the cells of the image2 which overlap the cell of the image1
        //  - the number of pixels overlaped
        //  - specify for each overlaped cell of image2 if the centroid is in the cell of image1
        // The elements of overlap are of a class (a little like a type (ie double, int, etc)) i created : couple

        if (debugMode)
        {
            std::cout << "\tOverlaps #" << firstImage + currentFrame - 2 << " > #" << firstImage + currentFrame - 1 << " ... " << std::endl;
        }

        // Cell - cell overlap calculation --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        std::map<int, std::vector<Couple> > overlap;
        Couple overlapFiller;
        OverlapCell2CellComparison(overlap, overlapFiller, image1, image2, pivUImage, pivVImage, imageSize);

        // Classification of the overlaping cells by decreasing overlap area ----------------------------------------------------------------------------------------------------------------------------------------------------
        std::map<int, std::vector<Couple> > bestOverlap;
        std::map<int, std::vector<Couple> > bestOverlapWCentroids;
        for (unsigned long currentCell = 1; currentCell <= numberOfCells1; ++currentCell)
        {
            // sort elements by number (lower first)
            sort(overlap[currentCell].begin(), overlap[currentCell].end(), CompareByNumber);
            // transfer overlap contents in bestOverlap
            while (!overlap[currentCell].empty())
            {
                bestOverlap[currentCell].push_back(overlap[currentCell].back());
                overlap[currentCell].pop_back();
            }
        }

        // Previous frame centroids association with cell overlap ---------------------------------------------------------------------------------------------------------------------------------------------------------------
        std::pair<double, double> *centroidsArray = new std::pair<double, double>[numberOfCells2];
        CentroidToCellOverlap(centroidsArray, numberOfCells2, bestOverlap, geometryFilter1, geometryFilter2, iterator1, iterator2, iteratorUPIV, iteratorVPIV, imageSize);

        // Sorting best overlap/centroids combination in decreasing order -------------------------------------------------------------------------------------------------------------------------------------------------------
        for (std::map<int, std::vector<Couple> >::iterator itBest = bestOverlap.begin(); itBest != bestOverlap.end(); ++itBest)
        {
            for (std::vector<Couple>::iterator itCple = itBest->second.begin(); itCple != itBest->second.end(); ++itCple)
            {
                if (itCple->GetCentroidIsInTheCellN())
                {
                    bestOverlapWCentroids[itBest->first].push_back(*itCple);
                }
            }
        }

        // Cell anisotropy map construction -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        std::map<unsigned long, double> anisotropy;
        for (unsigned long currentCell = 1; currentCell <= numberOfCells2; ++currentCell)
        {
            anisotropy[currentCell] = 1.0 - (geometryFilter2->GetMinorAxisLength(currentCell) / geometryFilter2->GetMajorAxisLength(currentCell)); // (2012-05-04)
        }

        // ---------- CORRESPONDANCE CALCULATION --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        if (debugMode)
        {
            std::cout << "\tCorrespondances ... " << std::endl;
        }

        // debug log
        std::ofstream dlog;
        if (debugMode)
        {
            dlog.open((outputPath + "/debugLOG_" + intToString(firstImage + currentFrame - 1) + ".txt").c_str());
        }

        // iterate on all cell in image1
        for (unsigned long currentCell = 1; currentCell <= numberOfCells1; currentCell++)
        {
            if (debugMode)
            {
                dlog << "= = = = = " << currentCell << " = = = = =" << std::endl;
            }

            // is cell in image1 coalesced?
            bool nCoalescedCell = (correspondance[currentFrame - 1][currentCell].size() > 1);
            double c1x = geometryFilter1->GetCentroid(currentCell)[0] < 0 ? -geometryFilter1->GetCentroid(currentCell)[0] : geometryFilter1->GetCentroid(currentCell)[0];
            double c1y = geometryFilter1->GetCentroid(currentCell)[1] < 0 ? -geometryFilter1->GetCentroid(currentCell)[1] : geometryFilter1->GetCentroid(currentCell)[1];

            // number of centroids of cells of image2 contained in the cell of image1
            int numberOfCentroids = bestOverlapWCentroids[currentCell].size();

            if (!nCoalescedCell) // non-coalesced case
            {
                if (debugMode)
                {
                    dlog << "not coalesced : " << numberOfCentroids << " overlapping centroids" << std::endl;
                }

                // if no centroid detected in the current cell, we search for other close centroids -----------------------------------------------------------------------------------------------------------------------------
                if (numberOfCentroids == 0)
                {
                    // centroids in the neighbourhood
                    std::vector<std::pair<unsigned long, double> > neighboringCentroids;

                    // look for centroids close enough
                    for (unsigned long otherCell = 0; otherCell < numberOfCells2; otherCell++)
                    {
                        // distance inter-centroid
                        double c2x = centroidsArray[otherCell].first;
                        double c2y = centroidsArray[otherCell].second;
                        double d = (c2x - c1x) * (c2x - c1x) + (c2y - c1y) * (c2y - c1y);

                        // only consider centroid within range
                        if (d < ExtendedCentroidRadius)
                        {
                            ITKImageType::IndexType mIndex;
                            ITKImageType::IndexType cIndex;
                            cIndex[0] = roundToInt(c2x);
                            cIndex[1] = roundToInt(c2y);

                            mIndex = ShiftPointByPIV(cIndex, iteratorUPIV, iteratorVPIV, imageSize, false);

                            // define local region around the moved centroid
                            ITKImageType::RegionType region;
                            // region size=3x3
                            ITKImageType::SizeType size;
                            size[0] = size[1] = 3;
                            // set size
                            region.SetSize(size);
                            // set starting index
                            region.SetIndex(mIndex);
                            // initialize iterator
                            ConstNeighborhoodIteratorType it(radius, image1, region);

                            // if any matching value ...
                            if (it.GetPixel(offset1) == currentCell || it.GetPixel(offset2) == currentCell || it.GetPixel(offset3) == currentCell || it.GetPixel(offset4) == currentCell
                                || it.GetPixel(offset5) == currentCell || it.GetPixel(offset6) == currentCell || it.GetPixel(offset7) == currentCell || it.GetPixel(offset8) == currentCell)
                            {
                                std::pair<unsigned long, double> neighborAssociation = std::make_pair(otherCell + 1, d);
                                neighboringCentroids.push_back(neighborAssociation);
                            }
                        } // end if centroid in range
                    } // end for loop over other cells

                    if (debugMode)
                    {
                        dlog << neighboringCentroids.size() << " suitable centroid(s) in the neighbourhood" << std::endl;
                    }

                    // determining the corresponding cell -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
                    unsigned long correspondingCell = 0;

                    if (neighboringCentroids.size() == 0)  // no centroid found in the neighbourhood
                    {
                        if (!bestOverlap[currentCell].empty()) // we associate the best overlap found
                        {
                            correspondingCell = bestOverlap[currentCell].front().GetPixelValue();
                        }
                    }
                    else if (neighboringCentroids.size() == 1) // one centroid found !
                    {
                        correspondingCell = neighboringCentroids.front().first;
                    }
                    else // two or more centroids found ...
                    {
                        double *overlapRatio = new double[neighboringCentroids.size()];
                        // for each detected centroids, we check the one with the best overlaping area
                        for (unsigned int currentNeighborCentroids = 0; currentNeighborCentroids < neighboringCentroids.size(); currentNeighborCentroids++)
                        {
                            bool found = false;
                            overlapRatio[currentNeighborCentroids] = 0;
                            for (std::vector<Couple>::iterator itCple = bestOverlap[currentCell].begin(); itCple != bestOverlap[currentCell].end() && !found; itCple++)
                            {
                                if (itCple->GetPixelValue() == neighboringCentroids[currentNeighborCentroids].first)
                                {
                                    // ratio  = overlap / cell area
                                    overlapRatio[currentNeighborCentroids] = itCple->GetNumber() / double(geometryFilter2->GetVolume(itCple->GetPixelValue()));
                                    found = true;
                                }
                            }
                        }

                        // find the max ratio
                        unsigned long bestRatio = 0;
                        for (unsigned int currentNeighborCentroids = 1; currentNeighborCentroids < neighboringCentroids.size(); currentNeighborCentroids++)
                        {
                            if (overlapRatio[currentNeighborCentroids] > overlapRatio[bestRatio])
                            {
                                bestRatio = currentNeighborCentroids;
                            }
                        }

                        // set the correspondingCell if ratio != 0
                        if (overlapRatio[bestRatio] != 0)
                        {
                            correspondingCell = neighboringCentroids[bestRatio].first;
                        }
                        else
                        {
                            std::cerr << "Error 'Main' : best overlap ratio equal to 0 while multiple centroids detected for region: " << currentCell << " ." << std::endl;
                        }

                        // deallocate memory
                        delete[] overlapRatio;
                    } // end if else for correspondingCell association

                    // fill the correspondance ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                    if (correspondingCell != 0)
                    {
                        correspondanceFiller.Delete();                                                                                      // clear filler
                        correspondanceFiller.SetAbsoluteNumber(correspondance[currentFrame - 1][currentCell].front().GetAbsoluteNumber());  // set Absolute number
                        correspondanceFiller.SetTag(correspondance[currentFrame - 1][currentCell].front().GetTag());                        // set Tag
                        correspondanceFiller.SetCentroid(0, c1x);                                                                           // set centroid X
                        correspondanceFiller.SetCentroid(1, c1y);                                                                           // set centroid Y
                        correspondance[currentFrame][correspondingCell].push_back(correspondanceFiller);                                    // fill the correspondance
                    }
                    else
                    {
                        std::cerr << "Warning 'Main' : no correspondance found for region: " << currentCell << " ." << std::endl;
                    }

                    if (debugMode)
                    {
                        dlog << currentCell << " -> " << correspondingCell << std::endl;
                    }
                } // end if no centroid detected

                // One centroid detected in the cell ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                else if (numberOfCentroids == 1)
                {
                    correspondanceFiller.Delete();                                                                                             // clear filler
                    correspondanceFiller.SetAbsoluteNumber(correspondance[currentFrame - 1][currentCell].front().GetAbsoluteNumber());         // set Absolute number
                    correspondanceFiller.SetTag(correspondance[currentFrame - 1][currentCell].front().GetTag());                               // set Tag
                    correspondanceFiller.SetCentroid(0, c1x);                                                                                  // set centroid X
                    correspondanceFiller.SetCentroid(1, c1y);                                                                                  // set centroid Y
                    correspondance[currentFrame][bestOverlapWCentroids[currentCell].front().GetPixelValue()].push_back(correspondanceFiller);  // fill the correspondance
                    if (debugMode)
                    {
                        dlog << currentCell << " -> " << bestOverlapWCentroids[currentCell].front().GetPixelValue() << std::endl;
                    }
                } // end if one centroids detected

                // Two centroids detected in the cell ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                else if (numberOfCentroids == 2)
                {
                    unsigned long firstCell = bestOverlapWCentroids[currentCell].front().GetPixelValue();
                    unsigned long secondCell = bestOverlapWCentroids[currentCell].back().GetPixelValue();

                    if (debugMode)
                    {
                        dlog << "division check: " << firstCell << " & " << secondCell << std::endl;
                    }

                    // Verification if it is a division
                    /*bool division = DivisionVerificationFunction(firstCell, secondCell, currentCell, neighbourMap, anisotropy, geometryFilter1, geometryFilter2,
                        borderCells1, borderCells2,
                        daughterDaughterRatio2, motherDaughterRatio2, anisotropyThreshold, debugMode, dlog);*/
                    bool division = DivisionVerificationFunction(firstCell, secondCell, currentCell, correspondance, neighbourMap, anisotropy, geometryFilter1, geometryFilter2, borderCells1, borderCells2, daughterDaughterRatio2, motherDaughterRatio2, anisotropyThreshold, maximumDivisionAllowed, currentFrame - 1, debugMode, dlog);

                    // a division is detected !
                    if (division)
                    {
                        if (debugMode)
                        {
                            dlog << "division accepted!\n" << currentCell << " => " << firstCell << " & " << secondCell << std::endl;
                        }

                        std::string tag1, tag2;

                        // first daughter -----------
                        correspondanceFiller.Delete();                                                                                      // clear filler
                        correspondanceFiller.SetAbsoluteNumber(correspondance[currentFrame - 1][currentCell].front().GetAbsoluteNumber());  // set Absolute number
                        correspondanceFiller.SetTag(correspondance[currentFrame - 1][currentCell].front().GetTag());                        // set Tag
                        correspondanceFiller.AddTag("1");                                                                                   // add divison tag
                        correspondanceFiller.setJustDivided();                                                                              // set division
                        correspondanceFiller.SetCentroid(0, c1x);                                                                           // set centroid X
                        correspondanceFiller.SetCentroid(1, c1y);                                                                           // set centroid Y
                        correspondance[currentFrame][firstCell].push_back(correspondanceFiller);                                            // fill the correspondance
                        tag1 = correspondanceFiller.getAbsoluteTag();
                        // second daughter ----------
                        correspondanceFiller.Delete();                                                                                      // clear filler
                        correspondanceFiller.SetAbsoluteNumber(correspondance[currentFrame - 1][currentCell].front().GetAbsoluteNumber());  // set Absolute number
                        correspondanceFiller.SetTag(correspondance[currentFrame - 1][currentCell].front().GetTag());                        // set Tag
                        correspondanceFiller.AddTag("2");                                                                                   // add divison tag
                        correspondanceFiller.setJustDivided();                                                                              // set division
                        correspondanceFiller.SetCentroid(0, c1x);                                                                           // set centroid X
                        correspondanceFiller.SetCentroid(1, c1y);                                                                           // set centroid Y
                        correspondance[currentFrame][secondCell].push_back(correspondanceFiller);                                           // fill the correspondance
                        tag2 = correspondanceFiller.getAbsoluteTag();

                        dividingCellsRN1.push_back(currentCell);                          // add to the dividing cells list
                        justDividedCellsRN2.push_back(std::make_pair(firstCell, secondCell));  // add to the just divided cells list
                        justDividedCellsAN2.push_back(std::make_pair(tag1, tag2));
                    }
                    else // division refused!
                    {
                        if (debugMode)
                        {
                            dlog << "division refused!\n" << currentCell << " -> " << firstCell << std::endl;
                        }
                        correspondanceFiller.Delete();                                                                                      // clear filler
                        correspondanceFiller.SetAbsoluteNumber(correspondance[currentFrame - 1][currentCell].front().GetAbsoluteNumber());  // set Absolute number
                        correspondanceFiller.SetTag(correspondance[currentFrame - 1][currentCell].front().GetTag());                        // set Tag
                        correspondanceFiller.SetCentroid(0, c1x);                                                                           // set centroid X
                        correspondanceFiller.SetCentroid(1, c1y);                                                                           // set centroid Y
                        correspondance[currentFrame][firstCell].push_back(correspondanceFiller);                                            // fill the correspondance
                    }
                } // end if two centroids detected in the cell

                // More than two centroids detected in the cell -----------------------------------------------------------------------------------------------------------------------------------------------------------------
                // only work on the three first centroids, others shall be ignored ...
                else
                {
                    bool found = false; // suitable couple found ?

                    if (debugMode)
                    {
                        dlog << "Looking for best couple ... " << std::endl;
                    }

                    // we check all possible couple
                    for (unsigned int centroids1 = 0; centroids1 < 2 && !found; centroids1++)
                    {
                        // first cell
                        unsigned long firstCell = bestOverlapWCentroids[currentCell][centroids1].GetPixelValue();
                        for (unsigned int centroids2 = centroids1 + 1; centroids2 < 3 && !found; centroids2++)
                        {
                            // second cell
                            unsigned long secondCell = bestOverlapWCentroids[currentCell][centroids2].GetPixelValue();

                            if (debugMode)
                            {
                                dlog << "division check: " << firstCell << " & " << secondCell << std::endl;
                            }

                            /*bool division = DivisionVerificationFunction(firstCell, secondCell, currentCell, neighbourMap, anisotropy, geometryFilter1, geometryFilter2,
                                borderCells1, borderCells2, daughterDaughterRatio2, motherDaughterRatio2, anisotropyThreshold, debugMode, dlog);*/
                            bool division = DivisionVerificationFunction(firstCell, secondCell, currentCell, correspondance, neighbourMap, anisotropy, geometryFilter1, geometryFilter2, borderCells1, borderCells2, daughterDaughterRatio2, motherDaughterRatio2, anisotropyThreshold, maximumDivisionAllowed, currentFrame - 1, debugMode, dlog);

                            // division detected !
                            if (division)
                            {
                                if (debugMode)
                                {
                                    dlog << "division accepted!\n" << currentCell << " => " << firstCell << " & " << secondCell << std::endl;
                                }

                                // suitable couple found, we stop the search
                                found = true;

                                std::string tag1, tag2;
                                // first daughter -----------
                                correspondanceFiller.Delete();                                                                                      // clear filler
                                correspondanceFiller.SetAbsoluteNumber(correspondance[currentFrame - 1][currentCell].front().GetAbsoluteNumber());  // set Absolute number
                                correspondanceFiller.SetTag(correspondance[currentFrame - 1][currentCell].front().GetTag());                        // set Tag
                                correspondanceFiller.AddTag("1");                                                                                   // add divison tag
                                correspondanceFiller.setJustDivided();                                                                              // set division
                                correspondanceFiller.SetCentroid(0, c1x);                                                                           // set centroid X
                                correspondanceFiller.SetCentroid(1, c1y);                                                                           // set centroid Y
                                correspondance[currentFrame][firstCell].push_back(correspondanceFiller);                                            // fill the correspondance
                                tag1 = correspondanceFiller.getAbsoluteTag();
                                // second daughter ----------
                                correspondanceFiller.Delete();                                                                                      // clear filler
                                correspondanceFiller.SetAbsoluteNumber(correspondance[currentFrame - 1][currentCell].front().GetAbsoluteNumber());  // set Absolute number
                                correspondanceFiller.SetTag(correspondance[currentFrame - 1][currentCell].front().GetTag());                        // set Tag
                                correspondanceFiller.AddTag("2");                                                                                   // add divison tag
                                correspondanceFiller.setJustDivided();                                                                              // set division
                                correspondanceFiller.SetCentroid(0, c1x);                                                                           // set centroid X
                                correspondanceFiller.SetCentroid(1, c1y);                                                                           // set centroid Y
                                correspondance[currentFrame][secondCell].push_back(correspondanceFiller);                                           // fill the correspondance
                                tag2 = correspondanceFiller.getAbsoluteTag();

                                dividingCellsRN1.push_back(currentCell);                          // add to the dividing cells list
                                justDividedCellsRN2.push_back(std::make_pair(firstCell, secondCell));  // add to the just divided cells list
                                justDividedCellsAN2.push_back(std::make_pair(tag1, tag2));
                            }
                        } // end of loop on centroid2
                    } // end of loop on centroid1

                    if (!found) // no suitable couple found
                    {
                        if (debugMode)
                        {
                            dlog << "all division refused!\n" << currentCell << " -> " << bestOverlapWCentroids[currentCell].front().GetPixelValue() << std::endl;
                        }

                        correspondanceFiller.Delete();                                                                                             // clear filler
                        correspondanceFiller.SetAbsoluteNumber(correspondance[currentFrame - 1][currentCell].front().GetAbsoluteNumber());         // set Absolute number
                        correspondanceFiller.SetTag(correspondance[currentFrame - 1][currentCell].front().GetTag());                               // set Tag
                        correspondanceFiller.SetCentroid(0, c1x);                                                                                  // set centroid X
                        correspondanceFiller.SetCentroid(1, c1y);                                                                                  // set centroid Y
                        correspondance[currentFrame][bestOverlapWCentroids[currentCell].front().GetPixelValue()].push_back(correspondanceFiller);  // fill the correspondance
                    }
                } // end if three centroids detected in cell
            } // end if not a coalescence
            else // décoalescence -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            {
                if (debugMode)
                {
                    dlog << "coalesced : trying decoalescence ..." << std::endl;
                }

                // pour tous les cellule overlapante (1ere boucle) on regarde toutes les cellules (2eboucle) qui sont coalescées dans la current cellule
                // on regarde si le centroide stocké est contenu dans la current overlapante (if(...)). si c'est le cas, on push_back le correspondance filler,
                // rempli avec les données de ce qui était stocké dans le correspondance [n-1], en particulier les coordonées du centroide stocké.

                ITKImageType::IndexType cIndex;
                for (std::vector<Couple>::iterator itCple = bestOverlap[currentCell].begin(); itCple != bestOverlap[currentCell].end(); itCple++)
                {
                    for (std::vector<CorrespondanceCouple>::iterator sa = correspondance[currentFrame - 1][currentCell].begin(); sa != correspondance[currentFrame - 1][currentCell].end(); sa++)
                    {
                        correspondanceFiller.Delete();                                    // reset filler
                        correspondanceFiller.SetAbsoluteNumber(sa->GetAbsoluteNumber());  // set Absolute number
                        correspondanceFiller.SetTag(sa->GetTag());                        // set Tag

                        // access to the centroid stored in the coalesced cell
                        cIndex[0] = roundToInt(sa->GetCentroid(0));
                        cIndex[1] = roundToInt(sa->GetCentroid(1));

                        ITKImageType::IndexType mIndex = ShiftPointByPIV(cIndex, iteratorUPIV, iteratorUPIV, imageSize, true);

                        correspondanceFiller.SetCentroid(0, mIndex[0]);  // set centroid X
                        correspondanceFiller.SetCentroid(1, mIndex[1]);  // set centroid Y

                        if (image2->GetPixel(mIndex) == itCple->GetPixelValue())
                        {
                            correspondance[currentFrame][itCple->GetPixelValue()].push_back(correspondanceFiller);
                            if (debugMode)
                            {
                                dlog << currentCell << " (" << correspondanceFiller.getAbsoluteTag_S() << ") -> " << itCple->GetPixelValue() << std::endl;
                            }
                        }
                    }
                }
            } //end of decoalescence
        } // end of correspondance filling

        // close debug log
        if (debugMode)
        {
            dlog.close();
        }

        // ---------- CORRECTION PATCH ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


        // correspondance patching ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if (debugMode)
        {
            std::cout << "\tCoalescences patch ... " << std::endl;
        }
        std::map<unsigned long, std::vector<std::string> > newlyCoalescedElements;
        unsigned long max = 0;
        CorrespondanceCorrectionPatch(&newlyCoalescedElements, max, correspondance, reverseTrack, borderCells2, newCellsRN_current, newCellsAN_current,
            centroidsArray, numberOfCells1, numberOfCells2, currentFrame, firstImage, outputPath, debugMode);
        // end of correspondance patching 


#if APOP_LAST  

        // blue patching --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if (correctionOption[0] == '1')
        {
            if (debugMode)
            {
                std::cout << "\tBlue patch ... " << std::endl;
            }
            BlueCorrectionPatch(newlyCoalescedElements, correspondance, neighbourMap, justDividedCellsRN2, justDividedCellsAN2, borderCells2, newCellsRN_current, newCellsAN_current, reverseTrack,
                currentFrame, firstImage, outputPath, debugMode);
        }

        //red - green patch ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if (correctionOption[1] == '1')
        {
            if (debugMode)
            {
                std::cout << "\tRed-Green patch ... " << std::endl;
            }
            RedGreenCorrectionPatch(newlyCoalescedElements, correspondance, reverseTrack, dividingCellsRN1, justDividedCellsRN2, justDividedCellsAN2, currentFrame, firstImage,
                outputPath, debugMode);
        }

        // red - green patch ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if (correctionOption[2] == '1')
        {
            if (debugMode)
            {
                std::cout << "\tRed-Green ext patch ... " << std::endl;
            }
            RedGreenExtendedCorrectionPatch(newlyCoalescedElements, correspondance, neighbourMap, borderCells2, reverseTrack, dividingCellsRN1, justDividedCellsRN2, justDividedCellsAN2, currentFrame, firstImage, outputPath, debugMode);
        }

        //  apoptosis patching --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if (correctionOption[3] == '1')
        {
            if (debugMode)
            {
                std::cout << "\tApoptosis patch ... " << std::endl;
            }
            ApoptosisPatch(newlyCoalescedElements, reverseTrack, labelsAreas, correspondance, geometryFilter1, apopSizePencentil, apoptosisCheckFrames,
                apoptosisThreshold, currentFrame, firstImage, outputPath, debugMode);
        }

#else

        // blue patching --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if (correctionOption[0] == '1')
        {
            if (debugMode)
            {
                std::cout << "\tBlue patch ... " << std::endl;
            }
            BlueCorrectionPatch(newlyCoalescedElements, correspondance, neighbourMap, justDividedCellsRN2, justDividedCellsAN2, borderCells2, newCellsRN_current, newCellsAN_current, reverseTrack,
                currentFrame, firstImage, outputPath, debugMode);
        }

        //  apoptosis patching --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if (correctionOption[1] == '1')
        {
            if (debugMode)
            {
                std::cout << "\tApoptosis patch ... " << std::endl;
            }
            ApoptosisPatch(newlyCoalescedElements, reverseTrack, labelsAreas, correspondance, geometryFilter1, apopSizePencentil, apoptosisCheckFrames,
                apoptosisThreshold, currentFrame, firstImage, outputPath, debugMode);
        }

        //red - green patch ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if (correctionOption[2] == '1')
        {
            if (debugMode)
            {
                std::cout << "\tRed-Green patch ... " << std::endl;
            }
            RedGreenCorrectionPatch(newlyCoalescedElements, correspondance, reverseTrack, dividingCellsRN1, justDividedCellsRN2, justDividedCellsAN2, currentFrame, firstImage,
                outputPath, debugMode);
        }

        // red - green patch ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if (correctionOption[3] == '1')
        {
            if (debugMode)
            {
                std::cout << "\tRed-Green ext patch ... " << std::endl;
            }
            RedGreenExtendedCorrectionPatch(newlyCoalescedElements, correspondance, neighbourMap, borderCells2, reverseTrack, dividingCellsRN1, justDividedCellsRN2, justDividedCellsAN2, currentFrame, firstImage, outputPath, debugMode);
        }

#endif


        // ---------- WRITING TRACKING OUTPUT -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if (debugMode)
        {
            std::cout << "\tPrinting files ... " << std::endl;
        }


        // New cells AN ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        std::ofstream newCellsAN((outputPath + "/new_cells_AN_" + intToString(firstImage + currentFrame - 1) + ".txt").c_str());
        if (newCellsAN_current.size() != 0)
        {
            for (std::vector<unsigned long>::iterator iteNewCell = newCellsAN_current.begin(); iteNewCell != newCellsAN_current.end(); iteNewCell++)
            {
                newCellsAN << *iteNewCell << std::endl;
            }
        }
        else // stores -1
        {
            newCellsAN << -1 << std::endl;
        }
        newCellsAN.close();

        // New cells RN ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        std::ofstream newCellsRN((outputPath + "/new_cells_RN_" + intToString(firstImage + currentFrame - 1) + ".txt").c_str());
        if (newCellsRN_current.size() != 0)
        {
            for (std::vector<unsigned long>::iterator iteNewCell = newCellsRN_current.begin(); iteNewCell != newCellsRN_current.end(); iteNewCell++)
            {
                newCellsRN << *iteNewCell << std::endl;
            }
        }
        else // stores -1
        {
            newCellsRN << -1 << std::endl;
        }
        newCellsRN.close();

        // Dividing cells -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        std::ofstream dividingCells((outputPath + "/dividing_cells_RN_" + intToString(firstImage + currentFrame - 2) + ".txt").c_str());
        if (dividingCellsRN1.size() != 0)
        {
            for (std::vector<unsigned long>::iterator iteDividingCell = dividingCellsRN1.begin(); iteDividingCell != dividingCellsRN1.end(); iteDividingCell++)
            {
                dividingCells << *iteDividingCell << std::endl;
            }
        }
        else // stores -1
        {
            dividingCells << -1 << std::endl;
        }
        dividingCells.close();

        // Just divided cells ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        std::ofstream justDividedCells((outputPath + "/just_divided_cells_RN_" + intToString(firstImage + currentFrame - 1) + ".txt").c_str());
        if (justDividedCellsRN2.size() != 0)
        {
            for (std::vector<std::pair<unsigned long, unsigned long> >::iterator iteJustDivided = justDividedCellsRN2.begin(); iteJustDivided != justDividedCellsRN2.end();
                iteJustDivided++)
            {
                justDividedCells << iteJustDivided->first << " " << iteJustDivided->second << std::endl;
            }
        }
        else
        {
            justDividedCells << -1 << std::endl;
        }
        justDividedCells.close();

        // Correspondance -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        std::ofstream corr((outputPath + "/correspondence_" + intToString(firstImage + currentFrame - 1) + ".txt").c_str()); // correspondance file
        std::ofstream corrbackup((outputPath + "/correspondance#_" + intToString(firstImage + currentFrame - 1) + ".txt").c_str()); // correspondance file for resume mode
        for (std::map<int, std::vector<CorrespondanceCouple> >::iterator itCorrespondance = correspondance[currentFrame].begin(); itCorrespondance != correspondance[currentFrame].end(); itCorrespondance++)
        {
            // check if coalesced
            if (itCorrespondance->second.size() > 1)
            {
                coalescedCellsRN2.push_back(itCorrespondance->first);
            }
            // print the cell correspondance
            for (std::vector<CorrespondanceCouple>::iterator iteCorr = itCorrespondance->second.begin(); iteCorr != itCorrespondance->second.end(); iteCorr++)
            {
                corr << itCorrespondance->first << " " << iteCorr->getAbsoluteTag() << std::endl;
                corrbackup << itCorrespondance->first << ' ' << iteCorr->getAbsoluteTag_S() << ' ' << iteCorr->GetCentroid(0) << ' ' << iteCorr->GetCentroid(1) << ' ' << iteCorr->isJustDivided() << std::endl;
                // update number of division at last frame
                if (nbDivision < iteCorr->GetTag().length() / 2)
                {
                    nbDivision = (unsigned short)iteCorr->GetTag().length() / 2;
                }
            }
        }
        corr.close();
        corrbackup.close();

        // Coalesced cells RN ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        std::ofstream coalescedCellsRN((outputPath + "/coalesced_cells_RN_" + intToString(firstImage + currentFrame - 1) + ".txt").c_str());
        if (coalescedCellsRN2.size() != 0)
        {
            for (std::vector<unsigned long>::iterator iteCoalescedCell = coalescedCellsRN2.begin(); iteCoalescedCell != coalescedCellsRN2.end(); iteCoalescedCell++)
            {
                coalescedCellsRN << *iteCoalescedCell << std::endl;
            }
        }
        else
        {
            coalescedCellsRN << -1 << std::endl;
        }
        coalescedCellsRN.close();


        // Reverse Correspondance -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if (debugMode)
        {
            std::ofstream rcorr((outputPath + "/rev_correspondance" + intToString(firstImage + currentFrame - 1) + ".txt").c_str());
            for (std::map<std::string, unsigned long >::iterator iteReversTrack = reverseTrack[currentFrame - 1].begin();
                iteReversTrack != reverseTrack[currentFrame - 1].end(); iteReversTrack++)
            {
                rcorr << iteReversTrack->first << "," << iteReversTrack->second << std::endl;
            }
            rcorr.close();
        }
    } //end of for MAIN loop on frames

    // ---------- END OF TRACKING PROCESS ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    // print the maximum number of division occured -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    std::ofstream mdivisionfile((outputPath + "/max_n_divisions_" + intToString(firstImage) + "-" + intToString(lastImage) + ".txt").c_str());
    mdivisionfile << nbDivision << std::endl;
    mdivisionfile.close();

    //bench end
    t_end = clock();
    // time elapsed
    double time_time = (t_end - t_start) / (double) CLOCKS_PER_SEC;
    std::cout << "Job Done in " << time_time << "sec." << std::endl;

    return EXIT_SUCCESS;
}

// ---------- HISTORIQUE --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// KNOWN BUGS:
// - When using correspondance[n].size() to get the number of cells in frame n, yields one too many line with respect to the number of cells in frame 2 of Scut 01
// - some best_overlap[i][j].GetPixelValue() have 0 as value instead of a proper region Relative Number (look up "decoalescence_test_" files). Could be due to membranes made up by 0s?
/*

* 2016-01-19: 4.0.1 (Stephane)
* Add maximum number of division allowed in parameter
* Division trying to go higher than maximum division are forbiden and should create a new cell instead

* 2015-12-07: 4.0 (Stephane)
* Important reshape of the code into separated function
* - reorganistaion of the global program structure (tracking.cpp, trackingincludes.h, trackingfunction.h, trackingfunction.cpp)
* - reorganisation of the tracking correction patches into Blue patch, Red-Green patch, Apoptosis patch, Red-Green extended patch
* - change of Red-Blue patch into Blue-patch, it loop on blue cell and search for red cell in their 1-neighbor and 2-neighbor
* - add of Red-Green extended, same as Red-Green but extended to 2-neighbor. It may create mistake that why it is done in last
* - move of Apoptosis patch after Blue and Red-Green patch in order to assume that the coalesced cell remaning are not simple assignement mistake
* - add new criteria in apopthosis detection, the last know cell area must be in the first X percentile of the cell size distribution
* Fixed bug from vector.erase(iterator) that was called inside the vector loop (<- fixed also MacOs compatibility and Debug compiling)
* Fixed nbDivision calculation
* Fixed bug from not updating just_divided_cells list in patches
* Fixed bug from not updating newlyCoalescedList in patches

* 2015-02-16: 3.6 (Stephane & Boris)
* - removed warnings (look for 3.6)
* - actually finally printing out "PIV_Grid.txt" file!!
* - updated "tools.cpp" to 1.1 by including "cmath" library

* 2012-09-07: 3.5 (Yûki)
* - CorrectionOption format changed. int -> std::string
* - Red-Green correction case added.
* - duplicate values in coalesced_cells_RN2 removed.
* - rglog_*, rblog_*, cellarea files generation limited to debugmode only.

* 2012-08-28: 3.4.1 (Yûki)
* - Red/Blue correction case corrected.
* - resolved_red_blue_*.txt content changed: now used as a process log file.
* - RB_solved_*.txt file generation added: replace resolved_red_blue_*.txt.
*
* 2012-07-12: 3.4 (Yûki)
* - Red/Blue correction case corrected.
* - image_n_cells map removed.
*
* 2012-06-25: 3.4- (Yûki)
* - Red/Blue correction case corrected.
* - Apoptosis correction case corrected.
* - parameters file use added.
*
* 2012-05-22: 3.3 (Yûki)
* - PIV_Grid.txt file generation added.
* - CppMT_parameters_*-*.txt file generation added.
* - max_n_division_*-*.txt file generatation added.
* - correspondence_*.txt file generation changed to frame by frame (instead of all in the end).
* - skeletonPixelsList_*.txt file generation added.
* - division filter 0 "daughters must be neighbours" added.
* - tracking resume mode added.
* - delaminating_cells_*.txt file format for empty frame changed.
* - apoptosis detection criterion changed.
* - resolved_red_blue_*.txt file generation added.
* - correspondance#_*.txt file generation added.
*
* 2012-03-06: 3.2.4 (Yûki)
* - Round replaced by roundToInt. (Round.cpp/h not used anymore)
* - some old debug code removed.
*
* 2012-03-05: 3.2.3 (Yûki)
* - solved coalescence removed from coalesced cell list.
* - jus_divided_cells list corrected if assignement corrected.
*
* 2012-02-28: 3.2.2 (Yûki)
* - blue cell/red cell bad assignement correction added.
* - test_decoalescence printing disabled unless in DEBUGMODE.
*
* 2012-02-21: 3.2 (Yûki)
* - cell neighbours information added.
* - generation of neighbours_RN_*.txt added.
*
* 2011-11-22: 3.1.4 (Yûki)
* - apoptosis output file changed to delaminating_cell_*.txt. (contains RN of dying cells)
* - just_divided_cells_RN_*.txt file format changed (daughter cells now on the same line).
*
* 2011-11-14: 3.1.3 (Boris)
* - updated "CorrespondanceCouple.cpp": added a " " between each tag added in AddTag so as division tags appear as "1 2" instead of "12"
* - accordingly removed " " before tag when building "correspondence"
*
* 2011-10-14: 3.1.2 (YUKI)
* - PIV file reading bug corrected (wrong frame number).
*
* 2011-07-27: 3.1.1 (YUKI)
* - security added in meanAreaValue computation to prevent division by 0.
*
* 2011-07-25: 3.1 (YUKI)
* - cell area txt file created.
* - PIV reading bug corrected (bad copy/paste error).
* - apoptosis detection added.(look for ~APOP)
* - generation of apoptosis_*.txt added.
* - generation of frame_*.txt removed.
*
* 2011-07-12: 3.0.2 (YUKI)
* - outOfBound case for pixels moved by PIV changed: set to borders instead of unmoved.
*
* 2011-07-08: 3.0.1 (BORIS)
* - added 1st line: #define _CRT_SECURE_NO_WARNINGS to remove warnings due to use of "itoa" function
* - moved program version to the top: std::string tracking_version = "..."
* - restored generation of txt file output for initial frame for coalesced cells, divided cells...
*
* 2011-07-05: 3.0
* - PIV added (based on Raphael's code from "tracking_24_08_piv_sans_errode.cxx"). (look for MODIF_PIV)
* - new argument added (for PIV files path)
* PIV filename must follow with the following structure : [name][_u_|_v_][number][.extension]
* for example: "p7_u_0001.txt" for the PIV u vector
* - itoa definition for non-Visual Studio compiler added.
* - itoa changed to _itoa to respect naming convention
* - benchmark added.
*
// 23/09/10: 2.5g
// - solved the Full Thorax bug: REPLACE ALL "unsigned short" (0 < values< 65 535) OCCURENCES BY "unsigned long" (0 < values < 4 294 967 295) IN MAIN AND IN ALL SUBROUTINES AS WELL.
//   Most likely, this was because the number of pixels was beyond 65 535 in the Full thorax.
// - Removed call to Int (similar to atoi('n'), using same trick as Char) that was not called by tracking anyway but was listed in CMakeList.

// 22/09/10: 2.5f
// - changed all calls to Char(n) by itoa(n,buffer,10) to avoid writing useless "char" txt files and possibly avoid problems related to this.
// - updated CMakeLists.txt and removed call to Char.cpp
// - specifies tracking version as first output

// 16/09/10: 2.5e
// - fixed decoalesce bug that was due to wrong storage of CoMs in the case "neighboringCentroids.size()==0" (used to be "geometryFilter2" instead of "geometryFilter1")
// - save "decoalescence_test_" txt file to check possible decoalescence mistakes (used to fix above bug)

// 15/09/10: 2.5d
// - calling input arguments in the main function instead of loading txt files: path, outputPath, firstImage, lastImage

// 10-13/09/10: 2.5c
// - saves with the right number of frame (not starting from 1 but from first_frame): changed:
// 1-> firstImage
// n -> firstImage + n-1
// - fixed bug when starting from a frame number n_start<10 and ending in n_end>10

// 09/09/10: 2.5b
// - performs initialization (frame 1) for border cells...
// - stores the number of cells in each frame now in map "image_n_cells"
// - loads firstImage and lastImage and derives numberOfFrames from it

// 08/09/10: correction of many bugs
// - set a=1 instead of a=0 (mistake) when determining best overlap with centroid: THIS WAS MESSING UP EVERYTHING!
// - changed numberOfCells2 into ..._1 when iterating over overlap[n-1]
// - removed copy/paste problem in Raph's new iteration to determine whether a centroid is in a cell/own cell
// - use of "Round" to get more accurate centroid coordinates
// - finished building up list of cells required for segmentation interface


// 07/09/10: version 2.5a
// - start adapting changes made in version 1.0 and 2.0a for Matlab interfacing
// - REmoved the NaN line in the 1st line
*/