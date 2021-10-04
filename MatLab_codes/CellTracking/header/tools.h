/*** tools.h
 *
 * Toolkit ...
 *
 * @author GOYA Yuki
 * @version 2012-03-06
 */

#ifndef __tools_h__
#define __tools_h__

#include <string>

//#define PI 3.14159265358979224

std::string doubleToString(double); //2012-05-04
std::string intToString(int);
std::string longToString(long);
std::string numberToString(unsigned int, unsigned int);
std::string chomp(std::string);
unsigned int indexToMatlabIndex(unsigned int, unsigned int, unsigned int);
std::pair<unsigned int, unsigned int> indexToXY(unsigned int, unsigned int);
void indexToXY(unsigned int, unsigned int, std::pair<unsigned int, unsigned int> &);
long roundToInt(double); //2012-03-06
std::string dirpath(std::string); //2012-05-11
void pathspliter(std::string, std::string &, std::string &);
unsigned long roundToUInt(double);

template <class T_Number>
std::string numberToString(const T_Number &t) ;

#endif // __tools_h__
