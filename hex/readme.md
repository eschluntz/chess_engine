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
- 2D vector to hold graph nodes
- Graph data structure (adjacency list) to calculate connectivity
  - each edge is also a node, so a winning board is detected by being able to find a path between the two edge nodes
