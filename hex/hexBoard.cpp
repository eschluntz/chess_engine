#include <iostream>
#include <vector>
#include <string>
#include "hexBoard.h"

using namespace std;

HexBoard::HexBoard(int size)
{
    _size = size;
}

int main()
{
    vector<string> msg {"Hello", "C++", "World", "from", "VS Code", "and the C++ extension!"};
    HexBoard b = HexBoard(5);
    for (const string& word : msg)
    {
        cout << word << " ";
    }
    cout << endl;
}
