#ifndef __MATFILE_H
#define __MATFILE_H

#include <string> /* For strcmp() */
#include <vector> /* For STL */

#include "mat.h"

#include "Cell.h"
#include "Junction.h"


void FillCategoriesStructMat(mxArray * _pstr, std::vector< std::string > & _names, std::vector < Cell > & _list, unsigned int * nbCategories)
{
    mwSize dims[1];
    dims[0] = 1;

    unsigned int coreRNs = nbCategories[0];
    unsigned int FLRNs = nbCategories[1];
    unsigned int borderRNs = nbCategories[2];
    unsigned int nonBorderRNs = coreRNs + FLRNs;

    // Declare pointers to empty matlab matrix container
    mxArray * pcore = mxCreateDoubleMatrix(coreRNs, 1, mxREAL);
    mxArray * pfl = mxCreateDoubleMatrix(FLRNs, 1, mxREAL);
    mxArray * pborder = mxCreateDoubleMatrix(borderRNs, 1, mxREAL);
    mxArray * pnonborder = mxCreateDoubleMatrix(nonBorderRNs, 1, mxREAL);


    // Declare pointers to beginning of matlab matrix container
    double * core = mxGetPr(pcore);
    double * fl = mxGetPr(pfl);
    double * border = mxGetPr(pborder);
    double * nonborder = mxGetPr(pnonborder);

    // Parallele iteration on each object of the list
    int i;
    int cptCore(0), cptFL(0), cptBorder(0), cptNonBorder(0);
//#pragma omp parallel for shared(cptCore,cptFL,cptBorder,cptNonBorder) // openMP
    for (i = 0; i < _list.size(); i++)
    {
        if (_list[i].GetCategory() == 0)
        {
            core[cptCore] = _list[i].GetLabel();
            nonborder[cptNonBorder] = _list[i].GetLabel();
            cptCore++;
            cptNonBorder++;
        }
        else if (_list[i].GetCategory() == 1)
        {
            fl[cptFL] = _list[i].GetLabel();
            nonborder[cptNonBorder] = _list[i].GetLabel();
            cptFL++;
            cptNonBorder++;
        }
        else // category == 2
        {
            border[cptBorder] = _list[i].GetLabel();
            cptBorder++;
        }
    }

    // Associate matlab container to a matlab structure field
    int name_field(0), idx(0);
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pborder));    // border cell
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pfl));        // first layer cell
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pcore));      // core cell
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pnonborder)); // non border cell

    // Destroy matlab matrix container
    mxDestroyArray(pcore);
    mxDestroyArray(pfl);
    mxDestroyArray(pborder);
    mxDestroyArray(pnonborder);
}



void FillCellsStructMat(mxArray * _pstr, std::vector< std::string > & _names, std::vector < Cell > & _list, int _nSignals)
{
    mwSize dims[1];
    dims[0] = _list.size();
    
    std::vector< mxArray * > vppolaritymode;
    std::vector< mxArray * > vpbgintensity;
    std::vector< mxArray * > vpcontourintensity;
    
    // Declare pointers to empty matlab matrix container
    mxArray * plabel            = mxCreateDoubleMatrix(dims[0], 1, mxREAL);
    mxArray * pcategory         = mxCreateDoubleMatrix(dims[0], 1, mxREAL);
    mxArray * parea             = mxCreateDoubleMatrix(dims[0], 1, mxREAL);
    mxArray * panisotropy       = mxCreateDoubleMatrix(dims[0], 1, mxREAL);
    mxArray * porientation      = mxCreateDoubleMatrix(dims[0], 1, mxREAL);
    mxArray * pperimeter        = mxCreateDoubleMatrix(dims[0], 1, mxREAL);
    mxArray * pchordlength      = mxCreateDoubleMatrix(dims[0], 1, mxREAL);
    mxArray * pchorddisorder    = mxCreateDoubleMatrix(dims[0], 1, mxREAL);
    mxArray * pianisotropy      = mxCreateDoubleMatrix(dims[0], 1, mxREAL);
    mxArray * piorientation     = mxCreateDoubleMatrix(dims[0], 1, mxREAL);
    mxArray * pvorientation     = mxCreateDoubleMatrix(dims[0], 1, mxREAL);
    mxArray * pvpolarity        = mxCreateDoubleMatrix(dims[0], 1, mxREAL);
    mxArray * pnvertices        = mxCreateDoubleMatrix(dims[0], 1, mxREAL);
    mxArray * pnneighbors       = mxCreateDoubleMatrix(dims[0], 1, mxREAL);
    mxArray * pnsides           = mxCreateDoubleMatrix(dims[0], 1, mxREAL);
    mxArray * pnlinks           = mxCreateDoubleMatrix(dims[0], 1, mxREAL);
    mxArray * pcentroid         = mxCreateDoubleMatrix(dims[0], 2, mxREAL);
    mxArray * pitensor          = mxCreateDoubleMatrix(dims[0], 4, mxREAL);
    mxArray * pmtensor          = mxCreateDoubleMatrix(dims[0], 4, mxREAL);
    mxArray * pvtensor          = mxCreateDoubleMatrix(dims[0], 4, mxREAL);
    mxArray * ppolaritymode     = mxCreateDoubleMatrix(dims[0], 5, mxREAL);
    for (int i = 0; i < _nSignals; i++)
    {
        vppolaritymode.push_back(mxCreateDoubleMatrix(dims[0], 5, mxREAL));
        vpcontourintensity.push_back(mxCreateDoubleMatrix(dims[0], 1, mxREAL));
        vpbgintensity.push_back(mxCreateDoubleMatrix(dims[0], 1, mxREAL));
    }
    
    // Declare pointers to empty matlab cell array container
    mxArray * pindices           = mxCreateCellArray(1, dims);
    mxArray * pcontourindices    = mxCreateCellArray(1, dims);
    mxArray * pdilcontourindices = mxCreateCellArray(1, dims);
    mxArray * pneighbors         = mxCreateCellArray(1, dims);
    mxArray * pvertices          = mxCreateCellArray(1, dims);
    mxArray * psides             = mxCreateCellArray(1, dims);
    
    // Declare pointers to beginning of matlab matrix container
    double * label            = mxGetPr(plabel);
    double * category         = mxGetPr(pcategory);
    double * area             = mxGetPr(parea);
    double * anisotropy       = mxGetPr(panisotropy);
    double * orientation      = mxGetPr(porientation);
    double * perimeter        = mxGetPr(pperimeter);
    double * chordlength      = mxGetPr(pchordlength);
    double * itensor          = mxGetPr(pitensor);
    double * mtensor          = mxGetPr(pmtensor);
    double * vtensor          = mxGetPr(pvtensor);
    double * centroid         = mxGetPr(pcentroid);
    double * chorddisorder    = mxGetPr(pchorddisorder);
    double * ianisotropy      = mxGetPr(pianisotropy);
    double * iorientation     = mxGetPr(piorientation);
    double * vorientation     = mxGetPr(pvorientation);
    double * vpolarity        = mxGetPr(pvpolarity);
    double * nvertices        = mxGetPr(pnvertices);
    double * nneighbors       = mxGetPr(pnneighbors);
    double * nsides           = mxGetPr(pnsides);
    double * nlinks           = mxGetPr(pnlinks);
    
    // Parallele iteration on each object of the list
    int i;
#pragma omp parallel for private(i) // openMP
    for ( i = 0; i < _list.size(); i++)
    {
        // Fill up matlab matrix with scalar
        label[i]         = _list[i].GetLabel();
        category[i]      = _list[i].GetCategory();
        area[i]          = _list[i].GetArea();
        anisotropy[i]    = _list[i].GetAnisotropy();
        orientation[i]   = _list[i].GetOrientation();
        perimeter[i]     = _list[i].GetPerimeter();
        chordlength[i]   = _list[i].GetChordLength();
        chorddisorder[i] = _list[i].GetChordDisorder();
        ianisotropy[i]   = _list[i].GetInertiaAnisotropy();
        iorientation[i]  = _list[i].GetInertiaOrientation();
        vorientation[i]  = _list[i].GetVertexOrientation();
        vpolarity[i]     = _list[i].GetVertexPolarity();
        nvertices[i]     = _list[i].GetVerticesList().size();
        nneighbors[i]    = _list[i].GetNeighborsList().size();
        nsides[i]        = _list[i].GetJunctionsList().size();
        nlinks[i]        = _list[i].GetNeighborsList().size();
        // Fill up matlab matrix with 2d vectors
        for (int j = i; j < (2 * dims[0]); j += dims[0])
        {
            int t = static_cast < int > (std::floor(j / dims[0]) + 1) % 2;
            centroid[j] = _list[i].GetCentroid()[t];  // +1 because matlab first idx = 1 // +1 add before because of scale multiplication
        }
        // Fill up matlab matrix with 4d vectors
        for (int j = i; j < (4 * dims[0]); j += dims[0])
        {
            int idx = static_cast <int> (std::floor(j / dims[0]));
            itensor[j] = _list[i].GetInertiaTensor()[idx];
            mtensor[j] = _list[i].GetTextureTensor()[idx];
            vtensor[j] = _list[i].GetVertexTensor() [idx];
        }
        // Fill up matlab matrix with undefined number of scalar
        for (int n = 0; n < _nSignals; n++)
        {
            double * polaritymode     = mxGetPr(vppolaritymode[n]);
            double * contourintensity = mxGetPr(vpcontourintensity[n]);
            double * bgintensity      = mxGetPr(vpbgintensity[n]);
            for (int j = i; j < (5 * dims[0]); j += dims[0])
            {
                polaritymode[j] = _list[i].GetPolarityMode(n)[static_cast < int > (std::floor(j / dims[0]))];
            }
            contourintensity[i] = _list[i].GetContourIntensityDisorder(n);
            bgintensity[i]      = _list[i].GetBackgroundIntensity(n);
        }
        
        // Fill up matlab temporary matrix for cell array
        std::vector < unsigned long > idxList;
        idxList = _list[i].GetIndicesPixelsList();
        mxArray * ptmp1 = mxCreateDoubleMatrix(idxList.size(), 1, mxREAL);
        double * tmp1 = mxGetPr(ptmp1);
        for (int j = 0; j < idxList.size(); j++)
        {
            tmp1[j] = idxList[j] + 1;
        }
        idxList = _list[i].GetJunctionsIndicesPixelsList();
        mxArray * ptmp2 = mxCreateDoubleMatrix(idxList.size(), 1, mxREAL);
        double * tmp2 = mxGetPr(ptmp2);
        for (int j = 0; j < idxList.size(); j++)
        {
            tmp2[j] = idxList[j] + 1;
        }
        idxList = _list[i].GetDilJunctionsIndicesPixelsList();
        mxArray * ptmp3 = mxCreateDoubleMatrix(idxList.size(), 1, mxREAL);
        double * tmp3 = mxGetPr(ptmp3);
        for (int j = 0; j < idxList.size(); j++)
        {
            tmp3[j] = idxList[j] + 1;
        }
        idxList = _list[i].GetNeighborsList();
        mxArray * ptmp4 = mxCreateDoubleMatrix(1, idxList.size(), mxREAL);
        double * tmp4 = mxGetPr(ptmp4);
        for (int j = 0; j < idxList.size(); j++)
        {
            tmp4[j] = idxList[j];
        }
        idxList = _list[i].GetVerticesList();
        mxArray * ptmp5 = mxCreateDoubleMatrix(1, idxList.size(), mxREAL);
        double * tmp5 = mxGetPr(ptmp5);
        for (int j = 0; j < idxList.size(); j++)
        {
            tmp5[j] = idxList[j] + 1;
        }
        idxList = _list[i].GetJunctionsList();
        mxArray * ptmp6 = mxCreateDoubleMatrix(idxList.size(), 1, mxREAL);
        double * tmp6 = mxGetPr(ptmp6);
        for (int j = 0; j < idxList.size(); j++)
        {
            tmp6[j] = idxList[j];
        }
        
        // Fill up matlab cell array with matlab matrix
        mxSetCell(pindices, i, mxDuplicateArray(ptmp1));
        mxSetCell(pcontourindices, i, mxDuplicateArray(ptmp2));
        mxSetCell(pdilcontourindices, i, mxDuplicateArray(ptmp3));
        mxSetCell(pneighbors, i, mxDuplicateArray(ptmp4));
        mxSetCell(pvertices, i, mxDuplicateArray(ptmp5));
        mxSetCell(psides, i, mxDuplicateArray(ptmp6));
        
        // Destroy temporary matlab matrix
        mxDestroyArray(ptmp1);
        mxDestroyArray(ptmp2);
        mxDestroyArray(ptmp3);
        mxDestroyArray(ptmp4);
        mxDestroyArray(ptmp5);
        mxDestroyArray(ptmp6);
    }
    
    // Associate matlab container to a matlab structure field
    int name_field (0), idx (0);
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(plabel));             // Numbers
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pcategory));          // Category
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pindices));           // Indices
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(parea));              // Areas
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pperimeter));         // Perimeters
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(panisotropy));        // Anisotropies
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(porientation));       // Orientations
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pcentroid));          // XYs
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pvertices));          // Vertices
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pnvertices));         // nVertices
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pneighbors));         // Neighbors
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pnneighbors));        // nNeighbors
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pcontourindices));    // ContourIndices
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pdilcontourindices)); // DilatedContourIndices
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pchordlength));       // ContourChordLengths
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(psides));             // Sides
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pnsides));            // nSides
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pchorddisorder));     // ChordDisorders
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pmtensor));           // Ms
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pnlinks));            // nLinks
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pitensor));           // Is
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pianisotropy));       // Ianisotropies
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(piorientation));      // Iorientations
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pvtensor));           // Vs
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pvpolarity));         // Vpolarities
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pvorientation));      // Vorientations
    for (int n = 0; n < _nSignals; n++)
    {
        name_field = mxGetFieldNumber(_pstr, _names[idx + (n * 3)].c_str());
        mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(vpcontourintensity[n]));  // SideIntensityDisorders
        name_field = mxGetFieldNumber(_pstr, _names[idx + 1 + (n * 3)].c_str());
        mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(vpbgintensity[n]));       // BGintensities
        name_field = mxGetFieldNumber(_pstr, _names[idx + 2 + (n * 3)].c_str());
        mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(vppolaritymode[n]));      // PolarityModes
    }
    
    // Destroy matlab matrix container
    mxDestroyArray(plabel);
    mxDestroyArray(pcategory);
    mxDestroyArray(parea);
    mxDestroyArray(panisotropy);
    mxDestroyArray(porientation);
    mxDestroyArray(pperimeter);
    mxDestroyArray(pchordlength);
    mxDestroyArray(pitensor);
    mxDestroyArray(pmtensor);
    mxDestroyArray(pvtensor);
    mxDestroyArray(ppolaritymode);
    mxDestroyArray(pcentroid);
    mxDestroyArray(pindices);
    mxDestroyArray(pcontourindices);
    mxDestroyArray(pdilcontourindices);
    mxDestroyArray(pneighbors);
    mxDestroyArray(pvertices);
    mxDestroyArray(psides);
    mxDestroyArray(pianisotropy);
    mxDestroyArray(piorientation);
    mxDestroyArray(pvpolarity);
    mxDestroyArray(pvorientation);
    mxDestroyArray(pnvertices);
    mxDestroyArray(pnneighbors);
    mxDestroyArray(pnsides);
    mxDestroyArray(pnlinks);
    for (int i = 0; i < _nSignals; i++)
    {
        mxDestroyArray(vppolaritymode[i]);
        mxDestroyArray(vpcontourintensity[i]);
        mxDestroyArray(vpbgintensity[i]);
    }
}

void FillSidesStructMat(mxArray * _pstr, std::vector< std::string > & _names, std::vector < Junction > & _list, int _nSignals)
{
    mwSize dims[1];
    dims[0] = _list.size();
    
    std::vector<mxArray *> vpContourIntensityList;
    std::vector<mxArray *> vpBGIntensityList;
    
    // Declare pointers to empty matlab matrix container
    mxArray * plabel            = mxCreateDoubleMatrix(dims[0], 1, mxREAL);
    mxArray * pchordlength      = mxCreateDoubleMatrix(dims[0], 1, mxREAL);
    //mxArray * pparts            = mxCreateDoubleMatrix(dims[0], 1, mxREAL);
    mxArray * pneighbors        = mxCreateDoubleMatrix(dims[0], 2, mxREAL);
    mxArray * pvertices         = mxCreateDoubleMatrix(dims[0], 2, mxREAL);
    mxArray * pvertexindices    = mxCreateDoubleMatrix(dims[0], 2, mxREAL);
    for (int i = 0; i < _nSignals; i++)
    {
        vpContourIntensityList.push_back(mxCreateDoubleMatrix(dims[0], 1, mxREAL));
        vpBGIntensityList.push_back(mxCreateDoubleMatrix(dims[0], 1, mxREAL));
    }
    
    // Declare pointers to empty matlab cell array container
    mxArray * pindices    = mxCreateCellArray(1, dims);
    mxArray * pdilindices = mxCreateCellArray(1, dims);
    
    // Declare pointers to beginning of matlab matrix container
    double * label            = mxGetPr(plabel);
    double * chordlength      = mxGetPr(pchordlength);
    double * neighbors        = mxGetPr(pneighbors);
    double * vertices         = mxGetPr(pvertices);
    double * vertexindices    = mxGetPr(pvertexindices);
    //double * parts            = mxGetPr(pparts);
    
    // Parallele iteration on each object of the list
    int i;
#pragma omp parallel for private(i) // openMP
    for (i = 0; i < _list.size(); i++)
    {
        // Fill up matlab matrix with scalar
        label[i]         = _list[i].GetLabel();
        chordlength[i]   = _list[i].GetChordLength();
        //parts[i]         = _list[i].GetParts();
        // Fill up matlab matrix with 2d vectors
        for (int j = i; j < (2 * dims[0]); j += dims[0])
        {
            neighbors[j] = NAN;
            vertices[j] = NAN;
            vertexindices[j] = NAN;
            unsigned int idx = static_cast <unsigned int> (std::floor(j / dims[0]));
            if (!_list[i].GetCellList().empty())
            {
                neighbors[j] = _list[i].GetNeighbor(idx);
            }
            if (!_list[i].GetVerticesList().empty())
            {
                vertices[j] = _list[i].GetVertex(idx) + 1;
                vertexindices[j] = _list[i].GetVertex(idx) + 1;
            }
        }
        // Fill up matlab matrix with undefined number of scalar
        for (int n = 0; n < _nSignals; n++)
        {
            double * contourintensity = mxGetPr(vpContourIntensityList[n]);
            double * bgIntensity      = mxGetPr(vpBGIntensityList[n]);
            contourintensity[i] = _list[i].GetJunctionIntensity(n);
            bgIntensity[i]      = _list[i].GetJunctionBackgroundIntensity(n);
        }
        
        // Fill up matlab temporary matrix for cell array
        std::vector < unsigned long > idxList;
        idxList = _list[i].GetIndicesPixelsList();
        mxArray * ptmp1 = mxCreateDoubleMatrix(1, idxList.size(), mxREAL);
        double * tmp1 = mxGetPr(ptmp1);
        for (int j = 0; j < idxList.size(); j++)
        {
            tmp1[j] = idxList[j] + 1;
        }
        idxList = _list[i].GetDilIndicesPixelsList();
        mxArray * ptmp2 = mxCreateDoubleMatrix(1, idxList.size(), mxREAL);
        double * tmp2 = mxGetPr(ptmp2);
        for (int j = 0; j < idxList.size(); j++)
        {
            tmp2[j] = idxList[j] + 1;
        }
        
        // Fill up matlab cell array with matlab matrix
        mxSetCell(pindices, i, mxDuplicateArray(ptmp1));
        mxSetCell(pdilindices, i, mxDuplicateArray(ptmp2));
        
        // Destroy temporary matlab matrix
        mxDestroyArray(ptmp1);
        mxDestroyArray(ptmp2);
    }
    
    // Associate matlab container to a matlab structure field
    int name_field (0), idx(0);
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(plabel));         // Numbers
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pneighbors));     // Cells
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pvertices));      // Vertices
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pvertexindices)); // VertexIndices
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pchordlength));   // ChordLength
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pindices));       // Indices
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pdilindices));    // DilatedIndices
    //name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    //mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pparts));         // Parts
    for (int n = 0; n < _nSignals; n++)
    {
        name_field = mxGetFieldNumber(_pstr, _names[idx + n].c_str());
        mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(vpContourIntensityList[n])); // Intensities
        name_field = mxGetFieldNumber(_pstr, _names[idx + 1 + n].c_str());
        mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(vpBGIntensityList[n]));      // BGintensities
    }
    
    // Destroy matlab matrix container
    mxDestroyArray(plabel);
    mxDestroyArray(pchordlength);
    mxDestroyArray(pindices);
    mxDestroyArray(pdilindices);
    mxDestroyArray(pneighbors);
    mxDestroyArray(pvertices);
    mxDestroyArray(pvertexindices);
    //mxDestroyArray(pparts);
    for (int i = 0; i < _nSignals; i++)
    {
        mxDestroyArray(vpContourIntensityList[i]);
        mxDestroyArray(vpBGIntensityList[i]);
    }
}

void FillVerticesStructMat(mxArray * _pstr, std::vector< std::string > & _names, std::vector <  Vertex > & _list)
{
    mwSize dims[1];
    dims[0] = _list.size();
    
    // Declare pointers to empty matlab matrix container
    mxArray * plabel      = mxCreateDoubleMatrix(dims[0], 1, mxREAL);
    mxArray * pindices    = mxCreateDoubleMatrix(dims[0], 1, mxREAL);
    mxArray * pnneighbors = mxCreateDoubleMatrix(dims[0], 1, mxREAL);
    mxArray * pcoord      = mxCreateDoubleMatrix(dims[0], 2, mxREAL);
    mxArray * pneighbors  = mxCreateDoubleMatrix(dims[0], 4, mxREAL);
    
    // Declare pointers to beginning of matlab matrix container
    double * label      = mxGetPr(plabel);
    double * indices    = mxGetPr(pindices);
    double * coord      = mxGetPr(pcoord);
    double * neighbors  = mxGetPr(pneighbors);
    double * nneighbors = mxGetPr(pnneighbors);
    
    // Parallele iteration on each object of the list
    int i;
#pragma omp parallel for private(i) // openMP
    for (i = 0; i < _list.size(); i++)
    {
        // Fill up matlab matrix with scalar
        label[i]      = _list[i].GetLabel() + 1;
        indices[i]    = _list[i].GetLinearIndex() + 1;
        nneighbors[i] = _list[i].GetCellList().size();
        // Fill up matlab matrix with 2d vectors
        for (int j = i; j < (2 * dims[0]); j += dims[0])
        {
            int t = static_cast < int > (std::floor(j / dims[0]) + 1) % 2;
            coord[j] = _list[i].GetCoordinate()[t] + 1;
        }
        // Fill up matlab matrix with 4d vectors
        std::vector < unsigned long > idxList = _list[i].GetCellList();
        for (int j = i; j < (4 * dims[0]); j += dims[0])
        {
            neighbors[j] = 0;
            if (static_cast < int > (std::floor(j / dims[0])) < idxList.size())
            {
                neighbors[j] = idxList[static_cast < int > (std::floor(j / dims[0]))];
            }
        }
    }
    
    // Associate matlab container to a matlab structure field
    int name_field (0), idx (0);
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(plabel));      // Numbers
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pindices));    // Indices
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pneighbors));  // Cells
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pnneighbors)); // nCells
    name_field = mxGetFieldNumber(_pstr, _names[idx++].c_str());
    mxSetFieldByNumber(_pstr, 0, name_field, mxDuplicateArray(pcoord));      // XYs
    
    // Destroy matlab matrix container
    mxDestroyArray(plabel);
    mxDestroyArray(pindices);
    mxDestroyArray(pneighbors);
    mxDestroyArray(pcoord);
}

#endif
