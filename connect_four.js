const readline = require('readline');
const EMPTY = 0;
const PLAYER = 1;
const OPPONENT = 2;
const ROWS = 6;
const COLS = 7;
const MAX_DEPTH = 4; // Depth limit for Minimax algorithm (hard mode)

let board = Array.from({ length: ROWS }, () => Array(COLS).fill(EMPTY));

const rl = readline.createInterface({
   input: process.stdin,
   output: process.stdout
});

function printBoard() {
   board.forEach(row => console.log(row.join(' ')));
   console.log('--------------------\n');
}

function checkWin(board, player) {
   for (let row = 0; row < ROWS; row++) {
       for (let col = 0; col < COLS; col++) {
           if (board[row][col] === player) {
               if (checkDirection(board, row, col, player, 1, 0) ||  // Horizontal
                   checkDirection(board, row, col, player, 0, 1) ||  // Vertical
                   checkDirection(board, row, col, player, 1, 1) ||  // Diagonal (top-left to bottom-right)
                   checkDirection(board, row, col, player, 1, -1)) { // Diagonal (bottom-left to top-right)
                   return player;
               }
           }
       }
   }
   return null;  // No winner yet
}

function checkDirection(board, row, col, player, dRow, dCol) {
   for (let i = 0; i < 4; i++) {
       let r = row + i * dRow;
       let c = col + i * dCol;
       if (r < 0 || r >= ROWS || c < 0 || c >= COLS || board[r][c] !== player) {
           return false;
       }
   }
   return true;
}

function isBoardFull(board) {
   return board[0].every(cell => cell !== EMPTY);
}


function generateSuccessors(board, player) {
   const successors = [];
   for (let col = 0; col < COLS; col++) {
       for (let row = ROWS - 1; row >= 0; row--) {
           if (board[row][col] === EMPTY) {
               let newBoard = JSON.parse(JSON.stringify(board));
               newBoard[row][col] = player;
               successors.push({ board: newBoard, move: col });
               break;
           }
       }
   }
   return successors;
}

// A* Search for Easy AI
function aStarSearch(board, player) {
   const openList = [];
   const closedList = [];
   const startNode = { board: board, move: null, g: 0, h: heuristicEasy(board), f: 0 };
   startNode.f = startNode.g + startNode.h;
   openList.push(startNode);

   while (openList.length > 0) {
       openList.sort((a, b) => a.f - b.f);
       const currentNode = openList.shift();
       closedList.push(currentNode);

       if (checkWin(currentNode.board, OPPONENT)) {
           return currentNode;
       }
       const successors = generateSuccessors(currentNode.board, player);
       for (let successor of successors) {
           if (closedList.find(node => JSON.stringify(node.board) === JSON.stringify(successor.board))) {
               continue;
           }

           successor.g = currentNode.g + 1;
           successor.f = successor.g;

           const existingNode = openList.find(node => JSON.stringify(node.board) === JSON.stringify(successor.board));
           if (!existingNode || successor.f < existingNode.f) {
               openList.push(successor);
           }
       }
   }
   return null;
}

// Hard Mode Heuristic and Minimax with Alpha-Beta Pruning
function evaluateLine(board, row, col, player) {
   let score = 0;
   const directions = [
       { dRow: 0, dCol: 1 }, // Horizontal
       { dRow: 1, dCol: 0 }, // Vertical
       { dRow: 1, dCol: 1 }, // Diagonal (top-left to bottom-right)
       { dRow: 1, dCol: -1 } // Diagonal (bottom-left to top-right)
   ];

   for (let { dRow, dCol } of directions) {
       let countPlayer = 0;
       let countOpponent = 0;
       let emptyCount = 0;
       for (let i = 0; i < 4; i++) {
           let r = row + i * dRow;
           let c = col + i * dCol;
           if (r >= 0 && r < ROWS && c >= 0 && c < COLS) {
               if (board[r][c] === player) {
                   countPlayer++;
               } 
               else if (board[r][c] === (player === PLAYER ? OPPONENT : PLAYER)) {
                   countOpponent++;
               } 
               else {
                   emptyCount++;
               }
           }
       }

       if (countPlayer === 3 && emptyCount === 1) {
           score += 100;
       } 
       else if (countOpponent === 3 && emptyCount === 1) {
           score -= 100;
       } 
       else if (countPlayer === 2 && emptyCount === 2) {
           score += 10;
       } 
       else if (countOpponent === 2 && emptyCount === 2) {
           score -= 10;
       }

       if (countPlayer === 1 && emptyCount === 3) {
           score += 5;
       } 
       else if (countOpponent === 1 && emptyCount === 3) {
           score -= 5;
       }
   }
   return score;
}

function heuristic(board) {
   let score = 0;
   for (let row = 0; row < ROWS; row++) {
       for (let col = 0; col < COLS; col++) {
           if (board[row][col] === OPPONENT) {
               score += evaluateLine(board, row, col, OPPONENT);
           } 
           else if (board[row][col] === PLAYER) {
               score -= evaluateLine(board, row, col, PLAYER);
           }
       }
   }
   return score;
}

function minimax(board, depth, alpha, beta, isMaximizingPlayer) {
   if (depth === 0 || checkWin(board, PLAYER) || checkWin(board, OPPONENT) || isBoardFull(board)) {
       return heuristic(board);
   }

   if (isMaximizingPlayer) {
       let maxEval = -Infinity;
       for (let col = 0; col < COLS; col++) {
           for (let row = ROWS - 1; row >= 0; row--) {
               if (board[row][col] === EMPTY) {
                   board[row][col] = OPPONENT;
                   let eval = minimax(board, depth - 1, alpha, beta, false);
                   board[row][col] = EMPTY;
                   maxEval = Math.max(maxEval, eval);
                   alpha = Math.max(alpha, eval);
                   if (beta <= alpha) break;
               }
           }
       }
       return maxEval;
   } 
   else {
       let minEval = Infinity;
       for (let col = 0; col < COLS; col++) {
           for (let row = ROWS - 1; row >= 0; row--) {
               if (board[row][col] === EMPTY) {
                   board[row][col] = PLAYER;
                   let eval = minimax(board, depth - 1, alpha, beta, true);
                   board[row][col] = EMPTY;
                   minEval = Math.min(minEval, eval);
                   beta = Math.min(beta, eval);
                   if (beta <= alpha) break;
               }
           }
       }
       return minEval;
   }
}

function bestMove(board, difficulty) {
   if (difficulty === 'easy') {
       const result = aStarSearch(board, OPPONENT);
       return result ? result.move : Math.floor(Math.random() * COLS);
   } 
   else {
       let bestScore = -Infinity;
       let move = -1;
       for (let col = 0; col < COLS; col++) {
           for (let row = ROWS - 1; row >= 0; row--) {
               if (board[row][col] === EMPTY) {
                   board[row][col] = OPPONENT;
                   let score = minimax(board, MAX_DEPTH, -Infinity, Infinity, false);
                   board[row][col] = EMPTY;
                   if (score > bestScore) {
                       bestScore = score;
                       move = col;
                   }
                   break;
               }
           }
       }
       return move;
   }
}

function placePiece(col, player) {
   for (let row = ROWS - 1; row >= 0; row--) {
       if (board[row][col] === EMPTY) {
           board[row][col] = player;
           break;
       }
   }
}

function startGame() {       
    console.log("The Current Board:");
    printBoard(); 
    playGame();
};


function playGame() {
   rl.question('Your move! Choose a column (0-6): ', (column) => {
       column = parseInt(column);

       if (column >= 0 && column < COLS) {
           placePiece(column, PLAYER);
           printBoard();

           if (checkWin(board, PLAYER) === PLAYER) {
               console.log("Congratulations! You won!");
               rl.close();
               return;
           }

           if (isBoardFull(board)) {
               console.log("It's a draw! The board is full.");
               rl.close();
               return;
           }

           const move = bestMove(board);
           console.log(`Opponent plays at column ${move}`);
           placePiece(move, OPPONENT);
           printBoard();

           if (checkWin(board, OPPONENT) === OPPONENT) {
               console.log("Opponent wins! Better luck next time.");
               rl.close();
               return;
           }

           if (isBoardFull(board)) {
               console.log("It's a draw! The board is full.");
               rl.close();
               return;
           }

           playGame();
       } 
       else {
           console.log("Invalid column, try again.");
           playGame();
       }
   });
}

startGame();