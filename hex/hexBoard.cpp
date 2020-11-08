#include <iostream>
#include <vector>
#include <sstream>
#include <string>
#include <limits>
#include <assert.h>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include "hexBoard.h"

using namespace std;


int HexBoard::rc_to_node(int row, int col)
/* converts row and column of grid into node_id */
{
    return col + _size * row;
}


HexBoard::HexBoard(int size)
/* Normal constructor */
{
    _size = size;
    int num_nodes = size * size + 4; // + 4 to include the 4 board sides as nodes
    _board_grid = vector<vector<char>>(size, vector<char> (size, '.'));
    _board_graph = vector<vector<int>>(num_nodes, vector<int> ()); // no edges yet
}

HexBoard::HexBoard(vector<vector<char>> board)
/* constructor which allows copying in an existing board */
{
    _size = board.size();
    _board_grid = board;
    // TODO init graph from a board

}

void HexBoard::draw()
/*
 * Draws the hexboard. Example 5x5 board with one move by X at position 1,1:
 * . — . — . — . — .
 *  \ / \ / \ / \ / \
 *   . — X — . — . — .
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


bool HexBoard::move(char player, int r, int c)
/*
 * Tries to make a legal move on the board.
 * Returns true if the move was legal.
 */
{
    // check if player is valid
    if (player != 'X' && player != 'O') {
        cout << "Warning: invalid player " << player << endl;
        return false;
    }
    // check if move is out of bounds
    if (r < 0 || c < 0 || r >= _size || c >= _size) {
        cout << "Warning: move (" << r << "," << c << ") is off the board" << endl;
        return false;
    }

    // check if move is already taken
    if (_board_grid[r][c] != '.' ) {
        cout << "Warning: move (" << r << "," << c << ") is already taken" << endl;
        return false;
    }

    // if we reach here, move is valid
    _board_grid[r][c] = player;
    // TODO update graph representation
    return true;
}


int getInt()
/*
 * Loops until it gets a proper integer from cin.
 * Adapted from: https://stackoverflow.com/questions/22442736/c-sanitize-integer-whole-number-input
 */
{
    int x = 0;
    while (!( cin >> x))
    {
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
        cout << "Please input a valid int: ";
     }

     cin.clear();
     cin.ignore(numeric_limits<streamsize>::max(), '\n');
     return (x);
 }


void HexBoard::get_user_move(char player)
/*
 * Prompts the player for a move until they enter a valid one.
 * makes that move.
 */
{
    draw();
    int row;
    int column;

    do {
        cout << "Please enter your move row: ";
        row = getInt();

        cout << "Please enter your move column: ";
        column = getInt();
    } while (!move(player, row, column));
}


void HexBoard::get_computer_move(char player)
/*
 * Does a random move for the computer.
 */
{
    int row, column;
    do {
        row = rand() % _size;
        column = rand() % _size;
    } while (!move(player, row, column));
}


bool HexBoard::is_over()
/*
 * Checks whether the game is over by trying to find a path between
 * either pair of sides using graph search.
 */
{
    return false;
}

/*
 * Unit tests
 */
void test_draw()
{
    // just make sure it doesn't segfault
    HexBoard b = HexBoard({
        {'.', 'X'},
        {'O', '.'},
    });
    b.draw();
}

void test_move()
{
    HexBoard b = HexBoard({
        {'.', 'X'},
        {'O', '.'},
    });

    assert(b.move('X', 0, 0) == true);
    assert(b.move('X', 0, 0) == false);
    assert(b.move('O', 0, 1) == false);
    assert(b.move('X', 0, -1) == false);
    assert(b.move('O', 0, 2) == false);
    assert(b.move('X', 2, 0) == false);
    assert(b.move('O', 1, 1) == true);
}

int main()
{
    test_draw();
    test_move();

    int size = 11;
    HexBoard b = HexBoard(size);

    cout << "Pick your side (X/O): ";
    char human = cin.get();
    human = toupper(human);
    if (human != 'X' && human != 'O') {
        cout << "Warning: invalid player. exiting.";
        return(-1);
    }

    // start game
    while (!b.is_over())
    {
        // X
        if (human == 'X') {
            b.get_user_move('X');
            b.get_computer_move('O');
        }
        else {
            b.get_computer_move('X');
            b.get_user_move('O');
        }
    }

    while (true)
        b.get_user_move('X');
}
