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
    HexBoard(vector<vector<char>> board);

    void draw();
    bool move(char player, int r, int c);
    void get_user_move(char player);
    void get_computer_move(char player);
    bool is_over();
};

#endif
