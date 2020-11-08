# Game of Hex

![board](https://upload.wikimedia.org/wikipedia/commons/3/38/Hex-board-11x11-%282%29.jpg)

https://en.wikipedia.org/wiki/Hex_(board_game)

## Plan
- [ ] graph representation of hex board
- [ ] display the board
- [ ] player chooses their color.
- [ ] player picks their move
- [ ] tell whether the move was legal or not
- [ ] simple AI to make a move
- [ ] check whether game is over

## Datastructure
- 2D vector to hold graph nodes
- Graph data structure (adjacency list) to calculate connectivity
  - each edge is also a node, so a winning board is detected by being able to find a path between the two edge nodes
