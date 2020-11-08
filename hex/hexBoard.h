#ifndef HEX_H
#define HEX_H

#include <iostream>
#include <vector>

using namespace std;

class HexBoard
{
private:
    int _size;
    vector<vector<char>> _board_grid;  //[R][C]
    // add graph

public:
    HexBoard(int size);

    void draw();
    bool move(int color, int x, int y);
    bool isOver();
};

#endif
