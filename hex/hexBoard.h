#ifndef HEX_H
#define HEX_H

#include <iostream>
#include <vector>

using namespace std;

class HexBoard
{
private:
    int _size;
    vector<vector<char>> _board_grid;  // [row][column] = 'X' or 'O'
    vector<vector<int>> _board_graph; // [node_id] -> vec(connected nodes)

public:
    HexBoard(int size);
    HexBoard(vector<vector<char>> board);

    void draw();
    bool move(char player, int r, int c);
    void get_user_move(char player);
    void get_computer_move(char player);
    int rc_to_node(int row, int col);
    bool is_over();
};

#endif
