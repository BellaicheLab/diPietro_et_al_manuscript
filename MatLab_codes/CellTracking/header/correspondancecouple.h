/*** correspondancecouple.h
 *
 * last update: 2012-03-05
 */

#ifndef __correspondancecouple_h__
#define __correspondancecouple_h__

//#include <iostream>
#include <string>

class CorrespondanceCouple
{
    std::string m_division_tag;
    unsigned long m_absolute_number;
    double m_x;
    double m_y;
    /*** boolean for newly born element (Yûki) */
    bool justDivided; // 2012-03-05

public:
    CorrespondanceCouple();
    void SetCentroid(int, double);
    void SetTag(std::string);
    void SetAbsoluteNumber(unsigned long);
    double GetCentroid(int);
    unsigned long GetAbsoluteNumber();
    std::string GetTag();
    std::string getAbsoluteTag();
    std::string getAbsoluteTag_S();
    void Delete();
    void AddTag(std::string);
    /*** true if newly born element (Yûki) */
    bool isJustDivided();
    /*** justDivided setter (Yûki) */
    void setJustDivided();
    std::string toString(); //not implemented yet
    void undivided();
};

#endif // __correspondancecouple_h__
