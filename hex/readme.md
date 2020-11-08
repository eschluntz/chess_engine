# Game of Hex

![board](https://upload.wikimedia.org/wikipedia/commons/3/38/Hex-board-11x11-%282%29.jpg)

https://en.wikipedia.org/wiki/Hex_(board_game)

## Plan
- [X] grid representation of hex board
- [X] display the board
- [X] player chooses their color.
- [X] player picks their move
- [X] tell whether the move was legal or not
- [X] simple AI to make a move
- [ ] graph representation of hex board
- [ ] check whether game is over

## Datastructure
- 2D vector grid to hold graph nodes
- Graph data structure (adjacency list) to calculate connectivity
  - the 4 edges of the game board are also nodes, so a winning board is detected by being able to find a path between the two edge nodes
  - node_id = column + _size * row
  - node_ids of top, right, bottom, left = _size * _size + [0-3]
  - vector <int (node_id)> -> vector of other node_ids it's connected to
