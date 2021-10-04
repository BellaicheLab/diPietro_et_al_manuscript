#include "correspondancecouple.h"

#include <sstream>
#include <algorithm>

using namespace std;

CorrespondanceCouple::CorrespondanceCouple()
{
    m_division_tag.clear();
    m_absolute_number = 0;
    m_x = 0;
    m_y = 0;
    // init justDivided (2012-03-05)
    justDivided = false;
}

void CorrespondanceCouple::Delete()
{
    m_division_tag = "";
    m_absolute_number = 0;
    m_x = 0;
    m_y = 0;
    // reset justDivided (2012-03-05)
    justDivided = false;
}

void CorrespondanceCouple::SetCentroid(int i, double a)
{
    if(i == 0)
        {
        m_x = a;
        }
    else if(i == 1)
        {
        m_y = a;
        }
}

void CorrespondanceCouple::SetTag(string b)
{
    m_division_tag = b;
}

double CorrespondanceCouple::GetCentroid(int j)
{
    if (j == 0)
        {
        return m_x;
        }
    else if (j == 1)
        {
        return m_y;
        }
    else
        {
        return 0;
        }
}

unsigned long CorrespondanceCouple::GetAbsoluteNumber()
{
    return m_absolute_number;
}

void CorrespondanceCouple::SetAbsoluteNumber(unsigned long c)
{
    m_absolute_number = c;
}

string CorrespondanceCouple::GetTag()
{
    return m_division_tag;
}

void CorrespondanceCouple::AddTag(string d)
{
    m_division_tag += " " + d;
}

// 2012-03-05: justDivided setter & getter (Yûki)
/*** justDivided setter */
void CorrespondanceCouple::setJustDivided()
{
    justDivided = true;
}
/*** access to justDivided
 * @return true if newly born element
 */
bool CorrespondanceCouple::isJustDivided()
{
    return justDivided;
}
string CorrespondanceCouple::getAbsoluteTag()
{
    ostringstream out;
    out << m_absolute_number;
    if(!m_division_tag.empty())
        {
        out << m_division_tag;
        }
    return out.str();
}
std::string CorrespondanceCouple::getAbsoluteTag_S()
{
    string s = this->getAbsoluteTag();
    replace(s.begin(), s.end(), ' ', '_');
    return s;
}

void CorrespondanceCouple::undivided()
{
    justDivided = false;
    m_division_tag.resize(m_division_tag.size() - 2); // remove the last division tag
}

/*** HISTORY ***/

/* 2012-03-05: (Yûki)
- justDivided attribute added.
*/

/* 14/11/2011: (Boris)
- added a " " between each tag added in AddTag so as division tags appear as "1 2" instead of "12"
*/