#include <iostream>
#include <vector>
#include <string>
#include "hexBoard.h"

using namespace std;

HexBoard::HexBoard(int size)
{
    _size = size;
    _board_grid = vector<vector<char>>(size, vector<char> (size, '.'));

    // fake data
    _board_grid[1][1] = 'R';
}

void HexBoard::draw()
/*
 * Draws the hexboard. Example 5x5 board with one move by Red at position 1,1:
 * . — . — . — . — .
 *  \ / \ / \ / \ / \
 *   . — R — . — . — .
 *    \ / \ / \ / \ / \
 *     . — . — . — . — .
 *      \ / \ / \ / \ / \
 *       . — . — . — . — .
 *        \ / \ / \ / \ / \
 *         . — . — . — . — .
 */
{
    string spacing("");  // increase this each loop and prepend the row with it
    for (int r = 0; r < _size; ++r) {
        // draw data
        cout << spacing;
        for (int c = 0; c < _size; ++c) {
            cout << _board_grid[r][c];

            // don't print a dash on the last one
            if (c < _size - 1)
                cout << " — ";
        }
        cout << endl;

        // draw connection lines
        if (r < _size - 1) {
            cout << spacing << " \\";
            for (int c = 0; c < _size - 1; ++c) {  // don't print lines after the last row
                cout << " / \\";
            }
            cout << endl;
            spacing += "  ";
        }

    }
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

    b.draw();
}
