# Tic-Tac-Toe Q-Learning

A complete implementation of tabular Q-learning for Tic-Tac-Toe with a Pygame GUI and comprehensive debugging features.

## Features

- **Q-Learning Agent**: Tabular Q-learning (no neural networks) with epsilon-greedy exploration
- **Pygame GUI**: Interactive 3×3 grid with visual feedback
- **Debug Panel**: Real-time display of state, valid actions, and Q-values
- **Two Opponent Modes**: Play against random or be the human opponent
- **Persistent Q-Table**: Save/load trained Q-table to JSON
- **Live Training**: Retrain the agent while the GUI is running

## Installation

### Option 1: Quick Setup (Windows)

```bash
# Run the setup script to create venv and install dependencies
setup.bat

# Then run the game
run.bat
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Manual run:

```bash
# Activate venv first
venv\Scripts\activate

# Run the program
python tictactoe_qlearning_pygame.py
```

On first run, the agent will train for 50,000 episodes (takes ~30-60 seconds), then the GUI opens.

## Controls

- **R** - Reset game
- **M** - Toggle opponent mode (Random/Human)
- **S** - Save Q-table to `qtable.json`
- **T** - Train +5,000 additional episodes
- **Click** - Place O mark (when in Human mode and it's O's turn)

## How It Works

### Q-Learning

The agent (X) learns using the Q-learning update rule:

```
Q(s, a) ← Q(s, a) + α[r + γ max_a' Q(s', a') - Q(s, a)]
```

- **States**: 9-character strings like `"X.O...X.."` representing the board
- **Actions**: Integers 0–8 (row-major board positions)
- **Rewards**:
  - Win: +1.0
  - Loss: -1.0
  - Draw: +0.2
  - Illegal move: -1.0 (safety)

### Hyperparameters

- Episodes: 50,000
- Learning rate (α): 0.1
- Discount factor (γ): 0.95
- Epsilon decay: 0.9995 (from 1.0 to 0.01)

### Debug Panel

The right-side panel shows:

- Current state string
- List of valid actions
- Q-values for all valid actions (sorted)
- Best action (highlighted in yellow on the board)
- Opponent mode
- Q-table size

The same information is also printed to the console for each move.

## Project Structure

- `tictactoe_qlearning_pygame.py` - Complete implementation (single file)
- `qtable.json` - Saved Q-table (auto-generated)
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Training Details

- X (agent) always starts
- During training, O plays randomly
- Epsilon-greedy policy for exploration during training
- Greedy policy (epsilon=0) during GUI gameplay
- Q-table is a dictionary mapping `(state, action)` tuples to Q-values

## Tips

- After training, X should win or draw against random opponent most of the time
- Try Human mode to challenge the trained agent!
- Use T key to train more if the agent isn't performing well
- Yellow highlight shows which cell X will choose (its best action)
