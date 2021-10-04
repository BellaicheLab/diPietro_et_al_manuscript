/*** tools.h
 *
 * Toolkit ...
 *
 * @author GOYA Yuki, Stephane
 * @version 2015-02-16
 */

#include "tools.h"

#include <sstream>
#include <cmath>    // 1.1

using namespace std;

/*** convert an int to a string
 * @param i the int to convert
 * @return the string representation of i
 */
string intToString(int i)
{
    ostringstream out;
    out << i;
    return out.str();
}
/*** convert an int to a string
 * @param i the long int to convert
 * @return the string representation of i
 */
string longToString(long i)
{
    ostringstream out;
    out << i;
    return out.str();
}

/*** convert a double to a string
 * @param d the double to convert
 * @return the string representation of d
 */
string doubleToString(double d)
{
    ostringstream out;
    out << d;
    return out.str();
}

/*** convert an int to a string with a specific number of digit
 * @param i the int to convert
 * @param ndigit the number of digit
 * @return the representation of i with ndigit digits.
 */
string numberToString(unsigned int i, unsigned int ndigit)
{
    string s_i = intToString(i);
    unsigned int padding = ndigit - s_i.size();
    while(padding-- != 0)
        {
        s_i = "0" + s_i;
        }
    return s_i;
}

/*** chop the trailing white spaces and new lines of a string.
 * aiming to be similar to Perl's chomp function ...
 * @param the string to chop
 * @return the chopped string
 */
string chomp(string s)
{
    int i = s.size() - 1;
    while(i >= 0 && (s[i] == '\n' || s[i] == ' ' || s[i] == '\r'))
        {
        i--;
        }
    return s.substr(0, i + 1);
}

/*** convert the pixel index (used within this program) into its Matlab equivalent.
 * @param index the pixel index to convert
 * @param height the image height
 * @param width the image width
 * @return the equivalent pixel index in Matlab
 */
unsigned int indexToMatlabIndex(unsigned int index, unsigned int height, unsigned int width)
{
    unsigned int x_c, y_c;
    x_c = index % width;
    y_c = (index - x_c) / width;
    return x_c * height + y_c + 1;
}

/*** convert the pixel index into (X,Y) coordinates.
 * @param index the pixel index
 * @param width the image width
 * @return the (X,Y) coordinates of index.
 */
pair<unsigned int, unsigned int> indexToXY(unsigned int index, unsigned int width)
{
    unsigned int x = index % width;
    unsigned int y = (index - x) / width;
    return make_pair(x, y);
}

/*** convert the pixel index into (X,Y) coordinates. macro version
 */
void indexToXY(unsigned int index, unsigned int width, pair<unsigned int, unsigned int> &xy)
{
    xy.first = index % width;
    xy.second = (index - xy.first) / width;
}

// 2012-03-06
/*** round to the nearest integer value
 * @param a the double to round
 * @return the nearest integer value
 */
long roundToInt(double a)
{
    return a < 0.0 ? (long)ceil(a - 0.5) : (long)floor(a + 0.5);
}

/*** round to the nearest integer value
 * @param a the double to round
 * @return the nearest integer value
 */
unsigned long roundToUInt(double a)
{
    return a < 0.0 ? (unsigned long)floor(abs(a) + 0.5) : (unsigned long)floor(a + 0.5);
}

//2012-05-11
string dirpath(string fullpath)
{
    size_t found = fullpath.find_last_of("/\\");
    return fullpath.substr(0, found + 1);
}

void pathspliter(string fullpath, string &folderpath, string &filename)
{
    size_t found = fullpath.find_last_of("/\\");
    folderpath = fullpath.substr(0, found + 1);
    filename = fullpath.substr(found + 1);
}

template <class T_Number>
string numberToString(const T_Number &num)
{
    ostringstream oss; // create a stream
    oss << num;              // insert value to stream
    return oss.str();      // return as a string
}

//end of file

/*

16/02/2015: 1.1
- included "cmath" library

*/

