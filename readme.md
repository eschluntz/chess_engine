[![Build Status](https://travis-ci.com/eschluntz/games.svg?branch=master)](https://travis-ci.com/eschluntz/games)

[![codecov](https://codecov.io/gh/eschluntz/games/branch/master/graph/badge.svg)](https://codecov.io/gh/eschluntz/games)

# Games!

A repo of simple AIs to play various games, and a testing ground for various testing, CI, and tooling that I'm interested in trying.

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

### Search Algorithm - Min-Max, Alpha-Beta pruning, Move ordering

Ideally I'll be able to use the same search function for both tic tac toe, chess, and future adversarial games!

1. Min-Max: the bread and butter of adversarial games. Assume that I'll do my best possible move and the opponent will do their best possible mvoe.
2. Alpha-Beta pruning: Because of how min-max works, we can establish upper and lower bounds of our other options, and quit exploring a branch of the tree early if we know that it won't be chosen. It's pretty incredible how much of a speedup alpha-beta gave me!
3. move ordering: exploring moves from best to worst makes alpha beta pruning WAY more effective. For chess I should explore using extra calls to the evaluation function to sort the moves, and then go down them. (or maybe even some sort of shallower tree search first, to order the options.)

![times](https://github.com/eschluntz/games/blob/master/time_graph.png?raw=true)

### Board Evaluation Function

For tic tac toe, a board evaluation function is kind of silly - it's easy enough to just search all the way to the leaf nodes of the tree. Nevertheless, to keep my API the same between chess and tic tac toe, I use an evaluation function that just returns `-1000, 0, or 1000` depending on whether the game is won, lost, or neither.

There was some cool Numpy vectorization to make the evaluation function really fast!


# Chess

### Board Representation

I want to stick as close as I can to tic-tac-toe: a 2D numpy grid. I've seen some representations online use different letters, and then capital / lowercase to denote team.

It looks like there are also representations that that are "piece centric" rather than "square centric". I'm going to stick with board centric for now becuase it's more intuitive to me, but I can see that a piece centric model might be easier to iterate over.

It also looks like for high perforamnce there are things called "bitfields"

Beyond storing the data, the board object will need to:

1. iterate over possible moves
  a. jumping pieces
  b. sliding pieces
  c. TODO: special moves like castling, en passant, and promoting pawns
     NOTE: I just realized that castling and en passant require more information than the current state of the board!
2. do a move
3. undo a move, and thus remember past board states

I will use (0,0) as the top left corner of the board, even chess indexes "1" as the bottom of the board. I think this will be cleaner for internal representation, and I can create an interface for UI to transform "rank and file" to "row and column".

### Move Representation

I'll store moves as ((from_r, from_c), (to_r, to_c)).
For storing move history, I'll also need to store any captured pice, so that moves can be undone.
history = [(move, captured_piece), ...]

At some point I'll probably need to be able to export or import to a common chess notation like PGN or Algebraic Notation.

### Move Generation

The best way I can think to do this is:

1. search through the board to each of my pieces
2. ~~iterate through it's possible moves ~~
3. ~~if that move is legal (i.e. on board, doesn't land on our own piece), add it to my move list~~
4. order the moves in some better way?

Upon further thought, it will probably be easiest to generate possible moves at the same time as testing legality - especially for "sliding" pieces, that must stop at the first thing they hit.

### Future Improvements

1. Finish all special moves: en passant, promoting pawns, castling
    [x] ensure that move and board datastructures can support them
    [x] castle
    [ ] en passant
        [ ] move generation
        [ ] do move
        [ ] undo move
        [ ] human entry
        [ ] tests
    [ ] promotion
2. look for speedup opportunities, especially in move generation
    [X] [not actually faster] store store 2 piece sets, one for each side, instead of one and filtering it.
    [ ] larger index array, to make it easier to do bounds checking.
3. Iterative deepening to keep a constant time, rather than depth level. also to improve move ordering
4. Transposition Tables - Basically a hashtable for scores for any board position we've seen so far.
  a. Use this with iterative deepening to provide move orderings using depth-1 saves.
