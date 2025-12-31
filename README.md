# Tic-Tac-Toe Q-Learning

A complete implementation of tabular Q-learning for Tic-Tac-Toe with an interactive Pygame GUI designed to **visualize the Q-learning process**.

## Features

- **🎮 Three Game Modes**:
  - **Human vs Human**: Two players take turns
  - **Human vs AI**: Play as O against the trained Q-learning agent (X)
  - **Train AI (Visualized)**: Watch Q-learning in action with real-time training visualization
- **🧠 Q-Learning Agent**: Tabular Q-learning (no neural networks) with epsilon-greedy exploration
- **📊 Training Visualization**: See episodes, rewards, Q-values, and the learning process in real-time
- **🎨 Modern GUI**: Clean interface with interactive 3×3 grid and visual feedback
- **🔍 Debug Panel**: Real-time display of state, valid actions, Q-values, and best actions (in AI mode)
- **💾 Persistent Q-Table**: Save/load trained Q-table to JSON
- **⚡ Live Training**: Retrain the agent from within the GUI

## Installation

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows Command Prompt:
venv\Scripts\activate
# Windows PowerShell:
venv\Scripts\Activate.ps1
# Git Bash / Linux / macOS:
source venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Make sure venv is activated (you should see (venv) in your prompt)
venv\Scripts\activate    # Windows CMD
source venv/Scripts/activate    # Git Bash / Linux / macOS

# Run the program
python tictactoe_qlearning_pygame.py
```

### First Time Launch:

When you launch the program, you'll see a **main menu** with options:

1. **Human vs Human** - Play against another person
2. **Human vs AI** - Play as O against the trained agent (X)
3. **Train AI (Visualized)** - Train a new Q-table while watching the learning process
4. **Quit** - Exit the program

If no saved Q-table exists, you can train one using option 3. The visualization shows:

- Episode progress and epsilon (exploration rate)
- The board state after each episode
- Move-by-move rewards and outcomes
- Q-learning formula and hyperparameters

## Controls

### In Menu:

- **Click** or press **1-4** to select mode
- **ESC** - Quit

### During Human vs AI Mode:

- **Click** - Place O mark (your turn)
- **R** - Reset game
- **S** - Save Q-table to `qtable.json`
- **T** - Train +5,000 additional episodes (with visualization)
- **ESC** - Return to menu

### During Human vs Human Mode:

- **Click** - Place X or O mark (takes turns)
- **R** - Reset game
- **ESC** - Return to menu

### During Training:

- **ESC** - Stop training and return to menu

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

The right-side panel (in **Human vs AI mode only**) shows:

- **Current State**: Visual mini-grid representation
- **Valid Actions**: List of legal moves
- **Q-values**: All valid actions sorted by Q-value (color-coded)
  - 🟢 Green: High positive values (good moves)
  - 🔵 Blue: Positive values
  - 🔴 Red: Negative values (bad moves)
  - ⚪ Gray: Zero values
- **Best Action**: Highlighted in yellow on the main board
- **Mode Info**: Current game mode
- **Q-table Size**: Number of state-action pairs learned
- **Episodes Trained**: Total training episodes completed

Debug information is also printed to the console for each move.

## Training Visualization

When you select **"Train AI (Visualized)"**, you'll see:

1. **Episode Counter**: Shows progress (e.g., "Episode 5000 / 50000")
2. **Epsilon Value**: Current exploration rate (starts at 1.0, decays to 0.01)
3. **Board State**: The final board from the current episode
4. **Episode Summary**:
   - Total moves made
   - Last 6 moves with actions and rewards
   - Game outcome (X wins, O wins, or Draw)
   - Color-coded rewards
5. **Q-Learning Formula**: Shows the update rule and hyperparameters

The visualization updates every 50 episodes, allowing you to see how the agent learns over time!

## Project Structure

- [`tictactoe_qlearning_pygame.py`](tictactoe_qlearning_pygame.py) - Complete implementation (single file)
- [`qtable.json`](qtable.json) - Saved Q-table (auto-generated after training)
- [`requirements.txt`](requirements.txt) - Python dependencies (pygame-ce)
- [`venv/`](venv) - Virtual environment (not committed to git)
- [`README.md`](README.md) - This file

## Training Details

- **X (agent)** always starts and learns via Q-learning
- **O (opponent)** plays randomly during training
- **Epsilon-greedy policy** for exploration during training (ε starts at 1.0, decays to 0.01)
- **Greedy policy** (ε = 0) during GUI gameplay for optimal performance
- **Q-table** is a dictionary mapping `(state, action)` tuples to Q-values
- **Training is visualized** in real-time showing the learning process
- **50,000 episodes** takes approximately 30-60 seconds to complete

## Tips

- 🎯 **First time?** Choose "Train AI" from the menu to create a Q-table
- 🧪 **Watch it learn!** The training visualization shows how Q-values evolve
- 🎮 **Challenge the AI**: After training, play Human vs AI mode - the AI should rarely lose!
- 💪 **Improve performance**: Use the T key in Human vs AI mode to train more episodes
- 💡 **Yellow highlight**: Shows the AI's best action based on Q-values
- 🎨 **Human vs Human**: Great for teaching or just playing with a friend!
- 💾 **Save your progress**: Q-table is auto-saved after training and can be manually saved with S key
