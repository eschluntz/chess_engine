[![Build Status](https://travis-ci.com/eschluntz/games.svg?branch=master)](https://travis-ci.com/eschluntz/games)

[![codecov](https://codecov.io/gh/eschluntz/games/branch/master/graph/badge.svg)](https://codecov.io/gh/eschluntz/games)

![image](https://github.com/eschluntz/chess_engine/assets/1383982/2ad3d29e-f857-477c-b23c-3bcc5136f251)


# Chess Engine

### Board Representation

I want to stick as close as I can to tic-tac-toe: a 2D numpy grid. I've seen some representations online use different letters, and then capital / lowercase to denote team.

It looks like there are also representations that that are "piece centric" rather than "square centric". I'm going to stick with board centric for now becuase it's more intuitive to me, but I can see that a piece centric model might be easier to iterate over.

It also looks like for high perforamnce there are things called "bitfields"

Beyond storing the data, the board object will need to:

1. iterate over possible moves
  a. jumping pieces
  b. sliding pieces
  c. special moves like castling, en passant, and promoting pawns
2. do a move
3. undo a move, and thus remember past board states

I will use (0,0) as the top left corner of the board, even chess indexes "1" as the bottom of the board. I think this will be cleaner for internal representation, and I can create an interface for UI to transform "rank and file" to "row and column".

### Move Representation

I'll store moves as ((from_r, from_c), (to_r, to_c)).
For storing move history, I'll also need to store any captured pice, so that moves can be undone.
history = [(move, captured_piece), ...]

### Move Generation

1. search through the board to each of my pieces
2. iterate through its valid moves
3. order the moves to support alpha-beta pruning

### Search Algorithm - Min-Max, Alpha-Beta pruning, Move ordering

Ideally I'll be able to use the same search function for both tic tac toe, chess, and future adversarial games!

1. Min-Max: the bread and butter of adversarial games. Assume that I'll do my best possible move and the opponent will do their best possible mvoe.
2. Alpha-Beta pruning: Because of how min-max works, we can establish upper and lower bounds of our other options, and quit exploring a branch of the tree early if we know that it won't be chosen. It's pretty incredible how much of a speedup alpha-beta gave me!
3. move ordering: exploring moves from best to worst makes alpha beta pruning WAY more effective. For chess I should explore using extra calls to the evaluation function to sort the moves, and then go down them. (or maybe even some sort of shallower tree search first, to order the options.)

![times](https://github.com/eschluntz/games/blob/master/time_graph.png?raw=true)

### Board Evaluation Function

This is where the biggest heuristics come into play. For now I'm using a piece-value table that records how valuable it is to have a piece at any given place on the board. In the future this should be learned!


# Tic-tac-toe

Proving problem before tackling chess, with very similar requirements:

1. Board state representation
2. Search algorithm (min-max + alpha-beta pruning + other heuristics)
3. Board evaluation function

Given a board object that can enumerate possible next moves, the search algorithm goes down the tree looking alternatively at what our best move is, and what an opponent's best move is. If we reach a recursion depth limit, we use the board evaluation function as a heuristic to just that end point.

### Board Design

![board](https://github.com/eschluntz/games/blob/master/display.png?raw=true)

The underlying data representation was pretty straight forward - a 2D numpy grid of chars.

The API for doing and searching through moves is interesting - originally I planned to have
`do_move()` return a new separate board object which could be passed around and modified further. However, this made a lot of execution time be spent copying data around.

Instead, I made `do_move()` modify the board object in place, and added an `undo_move()` function which pops from a stack of past moves to revert the board. This allows a single board object to be used across the entire search tree pushing and popping moves, without any copying of data. This is a pattern I had seen in online chess engines


### Future Improvements

1. Finish all special moves: [x]en passant, [x]promoting pawns, [x]castling, [ ] prevent castling across check
2. Iterative deepening to keep a constant time, rather than depth level. also to improve move ordering
3. Transposition Tables - Basically a hashtable for scores for any board position we've seen so far. Use this with iterative deepening to provide move orderings using depth-1 saves.
