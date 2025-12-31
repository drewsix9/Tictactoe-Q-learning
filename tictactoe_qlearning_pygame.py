"""
Tic-Tac-Toe Q-Learning with Pygame GUI
=======================================
A complete implementation of tabular Q-learning for Tic-Tac-Toe with:
- Q-table (no neural networks)
- Pygame GUI for playing against trained agent
- Debug panel showing state, actions, Q-values
- Save/load Q-table functionality
"""

import pygame
import random
import json
import os
from typing import List, Tuple, Optional, Dict

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
EPISODES = 50000
ALPHA = 0.1          # Learning rate
GAMMA = 0.95         # Discount factor
EPSILON_START = 1.0  # Initial exploration rate
EPSILON_MIN = 0.01   # Minimum exploration rate
EPSILON_DECAY = 0.9995  # Decay per episode

# ============================================================================
# REWARDS
# ============================================================================
REWARD_WIN = 1.0
REWARD_LOSE = -1.0
REWARD_DRAW = 0.2
REWARD_ILLEGAL = -1.0

# ============================================================================
# FILE PATHS
# ============================================================================
Q_TABLE_FILE = "qtable.json"

# ============================================================================
# PYGAME SETTINGS
# ============================================================================
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 650
CELL_SIZE = 140
BOARD_OFFSET_X = 70
BOARD_OFFSET_Y = 120
GRID_COLOR = (40, 40, 40)
BG_COLOR = (250, 250, 252)
X_COLOR = (231, 76, 60)  # Softer red
O_COLOR = (52, 152, 219)  # Softer blue
TEXT_COLOR = (44, 62, 80)
HEADER_COLOR = (41, 128, 185)
HIGHLIGHT_COLOR = (255, 235, 59, 120)  # Yellow with alpha
PANEL_BG = (255, 255, 255)
PANEL_BORDER = (200, 200, 200)


# ============================================================================
# CORE TIC-TAC-TOE FUNCTIONS
# ============================================================================

def get_valid_actions(state: str) -> List[int]:
    """
    Return list of valid actions (indices 0-8) for cells that are empty ('.').
    """
    return [i for i in range(9) if state[i] == '.']


def apply_action(state: str, action: int, player: str) -> str:
    """
    Return a new state with the player's mark placed at the given action.
    Does NOT check validity - caller must ensure action is valid.
    """
    state_list = list(state)
    state_list[action] = player
    return ''.join(state_list)


def check_winner(state: str) -> Optional[str]:
    """
    Check if there's a winner or draw.
    Returns:
        'X' if X wins
        'O' if O wins
        'draw' if board is full and no winner
        None if game is still ongoing
    """
    # All possible winning lines (indices)
    lines = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # cols
        [0, 4, 8], [2, 4, 6]              # diagonals
    ]

    for line in lines:
        chars = [state[i] for i in line]
        if chars[0] == chars[1] == chars[2] and chars[0] in ['X', 'O']:
            return chars[0]

    # Check for draw
    if '.' not in state:
        return 'draw'

    return None


def choose_action(state: str, valid_actions: List[int], Q: Dict, epsilon: float) -> int:
    """
    Epsilon-greedy action selection.
    With probability epsilon, choose random action.
    Otherwise, choose action with highest Q-value.
    """
    if random.random() < epsilon:
        return random.choice(valid_actions)

    # Greedy: choose best action based on Q-values
    q_values = [Q.get((state, a), 0.0) for a in valid_actions]
    max_q = max(q_values)
    # Handle ties: randomly pick among actions with max Q-value
    best_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
    return random.choice(best_actions)


def q_update(Q: Dict, s: str, a: int, r: float, s_next: str,
             valid_actions_next: List[int], alpha: float, gamma: float):
    """
    Q-learning update rule (tabular):
    Q(s, a) ← Q(s, a) + α[r + γ max_a' Q(s', a') - Q(s, a)]

    If s_next is terminal (no valid actions), max Q is 0.
    """
    current_q = Q.get((s, a), 0.0)

    if len(valid_actions_next) == 0:
        # Terminal state
        max_q_next = 0.0
    else:
        max_q_next = max([Q.get((s_next, a_next), 0.0)
                         for a_next in valid_actions_next])

    new_q = current_q + alpha * (r + gamma * max_q_next - current_q)
    Q[(s, a)] = new_q


# ============================================================================
# TRAINING
# ============================================================================

def train(Q: Dict, episodes: int, alpha: float, gamma: float,
          epsilon_start: float, epsilon_min: float, epsilon_decay: float,
          visualize: bool = False, screen=None, fonts=None):
    """
    Train the Q-learning agent by playing against a random opponent.
    X (agent) always starts.
    If visualize=True, shows training progress in real-time with pygame.
    """
    epsilon = epsilon_start
    clock = pygame.time.Clock() if visualize else None

    for episode in range(episodes):
        state = '.' * 9  # Empty board
        episode_history = []  # Track state, action, reward for visualization

        # X always starts
        while True:
            # --- X's turn (agent) ---
            valid_actions = get_valid_actions(state)
            if len(valid_actions) == 0:
                break

            action_x = choose_action(state, valid_actions, Q, epsilon)
            state_x = state
            state = apply_action(state, action_x, 'X')

            # Check if X won
            result = check_winner(state)
            if result == 'X':
                reward = REWARD_WIN
                q_update(Q, state_x, action_x, reward, state, [], alpha, gamma)
                episode_history.append(
                    (state_x, action_x, reward, state, 'X wins!'))
                break
            elif result == 'draw':
                reward = REWARD_DRAW
                q_update(Q, state_x, action_x, reward, state, [], alpha, gamma)
                episode_history.append(
                    (state_x, action_x, reward, state, 'Draw'))
                break

            # --- O's turn (random opponent) ---
            valid_actions_o = get_valid_actions(state)
            if len(valid_actions_o) == 0:
                break

            action_o = random.choice(valid_actions_o)
            state_after_o = apply_action(state, action_o, 'O')

            # Check if O won
            result = check_winner(state_after_o)
            if result == 'O':
                # X loses
                reward = REWARD_LOSE
                q_update(Q, state_x, action_x, reward,
                         state_after_o, [], alpha, gamma)
                episode_history.append(
                    (state_x, action_x, reward, state_after_o, 'O wins'))
                state = state_after_o
                break
            elif result == 'draw':
                reward = REWARD_DRAW
                q_update(Q, state_x, action_x, reward,
                         state_after_o, [], alpha, gamma)
                episode_history.append(
                    (state_x, action_x, reward, state_after_o, 'Draw'))
                state = state_after_o
                break
            else:
                # Game continues: update Q for X's move with intermediate reward 0
                valid_actions_next = get_valid_actions(state_after_o)
                q_update(Q, state_x, action_x, 0.0, state_after_o,
                         valid_actions_next, alpha, gamma)
                episode_history.append(
                    (state_x, action_x, 0.0, state_after_o, 'Continue'))
                state = state_after_o

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Visualization - show every 50th episode
        if visualize and screen and fonts and episode % 50 == 0:
            # Check for quit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False  # Signal to stop training
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return False

            draw_training_visualization(screen, fonts, Q, episode, episodes, epsilon,
                                        episode_history, state)
            pygame.time.delay(200)  # Add 200ms delay for better visibility
            if clock:
                clock.tick(5)  # Limit to 5 FPS for slower visualization

        # Progress reporting
        if (episode + 1) % 5000 == 0:
            print(
                f"Episode {episode + 1}/{episodes}, epsilon={epsilon:.4f}, Q-table size={len(Q)}")

    print(f"Training complete! Q-table has {len(Q)} entries.")
    return True  # Training completed successfully


# ============================================================================
# Q-TABLE PERSISTENCE
# ============================================================================

def save_qtable(Q: Dict, filename: str):
    """Save Q-table to JSON file."""
    # Convert tuple keys to string keys for JSON
    Q_serializable = {f"{s}|{a}": v for (s, a), v in Q.items()}
    with open(filename, 'w') as f:
        json.dump(Q_serializable, f)
    print(f"Q-table saved to {filename} ({len(Q)} entries)")


def load_qtable(filename: str) -> Dict:
    """Load Q-table from JSON file."""
    if not os.path.exists(filename):
        return {}

    with open(filename, 'r') as f:
        Q_serializable = json.load(f)

    # Convert string keys back to tuple keys
    Q = {}
    for key, value in Q_serializable.items():
        s, a = key.rsplit('|', 1)
        Q[(s, int(a))] = value

    print(f"Q-table loaded from {filename} ({len(Q)} entries)")
    return Q


# ============================================================================
# DEBUG INFO
# ============================================================================

def get_best_action(state: str, valid_actions: List[int], Q: Dict) -> int:
    """Get the greedy (best) action for the current state."""
    if not valid_actions:
        return -1
    q_values = [Q.get((state, a), 0.0) for a in valid_actions]
    max_q = max(q_values)
    best_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
    return best_actions[0]


def get_q_values_sorted(state: str, valid_actions: List[int], Q: Dict) -> List[Tuple[int, float]]:
    """Return list of (action, q_value) sorted by Q-value descending."""
    q_list = [(a, Q.get((state, a), 0.0)) for a in valid_actions]
    q_list.sort(key=lambda x: x[1], reverse=True)
    return q_list


def print_debug_info(state: str, Q: Dict):
    """Print debug information to console."""
    valid_actions = get_valid_actions(state)
    print(f"\n--- DEBUG INFO ---")
    print(f"State: {state}")
    print(f"Valid actions: {valid_actions}")

    if valid_actions:
        q_sorted = get_q_values_sorted(state, valid_actions, Q)
        print("Q-values (sorted):")
        for a, q in q_sorted:
            print(f"  a={a} -> Q={q:.4f}")
        best = get_best_action(state, valid_actions, Q)
        print(f"Best action (greedy): {best}")
    else:
        print("No valid actions (terminal state)")
    print("------------------\n")


# ============================================================================
# PYGAME GUI
# ============================================================================

def run_game(Q: Dict, mode: str = 'human_vs_ai'):
    """
    Run the Pygame GUI for playing Tic-Tac-Toe.
    Modes:
        'human_vs_ai': X is AI, O is human
        'human_vs_human': Both X and O are human players
    """
    screen = pygame.display.get_surface()  # Use existing screen
    clock = pygame.time.Clock()

    # Fonts
    font_large = pygame.font.Font(None, 80)
    font_medium = pygame.font.Font(None, 36)
    font_small = pygame.font.Font(None, 24)

    # Game state
    state = '.' * 9
    game_over = False
    winner = None

    # Determine initial message based on mode
    if mode == 'human_vs_human':
        message = "X's turn (Human)"
        current_player = 'X'
    else:  # human_vs_ai
        message = "X (AI) vs O (Human)"
        current_player = 'X'

    # UI state
    episode_count = EPISODES

    running = True
    while running:
        screen.fill(BG_COLOR)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                # Save Q-table
                if event.key == pygame.K_s:
                    save_qtable(Q, Q_TABLE_FILE)
                    message = "Q-table saved!"

                # Retrain (only in AI mode)
                elif event.key == pygame.K_t and mode == 'human_vs_ai':
                    message = "Training 5000 episodes..."
                    pygame.display.flip()
                    fonts = (font_large, font_medium, font_small)
                    train(Q, 5000, ALPHA, GAMMA, EPSILON_MIN, EPSILON_MIN, 1.0,
                          visualize=True, screen=screen, fonts=fonts)
                    episode_count += 5000
                    message = f"Training complete! Total: ~{episode_count} episodes"

                # Reset game
                elif event.key == pygame.K_r:
                    state = '.' * 9
                    game_over = False
                    winner = None
                    if mode == 'human_vs_human':
                        message = "X's turn (Human)"
                        current_player = 'X'
                    else:
                        message = "X (AI) vs O (Human)"

                # ESC to return to menu
                elif event.key == pygame.K_ESCAPE:
                    running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                # Handle human click
                mouse_x, mouse_y = event.pos
                # Check if click is on board
                board_x = mouse_x - BOARD_OFFSET_X
                board_y = mouse_y - BOARD_OFFSET_Y

                if 0 <= board_x < CELL_SIZE * 3 and 0 <= board_y < CELL_SIZE * 3:
                    col = board_x // CELL_SIZE
                    row = board_y // CELL_SIZE
                    action = row * 3 + col

                    current_turn = get_current_turn(state)

                    # In human vs human: allow both players
                    # In human vs AI: only allow human (O) to click
                    can_play = False
                    if mode == 'human_vs_human':
                        can_play = action in get_valid_actions(state)
                        player = current_turn
                    elif mode == 'human_vs_ai' and current_turn == 'O':
                        can_play = action in get_valid_actions(state)
                        player = 'O'

                    if can_play:
                        state = apply_action(state, action, player)
                        print_debug_info(state, Q)

                        # Check game over
                        result = check_winner(state)
                        if result:
                            game_over = True
                            winner = result
                            if result == 'X':
                                message = "X WINS!"
                            elif result == 'O':
                                message = "O WINS!"
                            elif result == 'draw':
                                message = "DRAW!"
                        else:
                            # Update message
                            next_turn = get_current_turn(state)
                            if mode == 'human_vs_human':
                                message = f"{next_turn}'s turn"
                            else:
                                message = "X (AI) thinking..." if next_turn == 'X' else "O's turn"

        # AI logic: X moves automatically in human_vs_ai mode
        if not game_over and mode == 'human_vs_ai':
            current_turn = get_current_turn(state)

            if current_turn == 'X':
                valid_actions = get_valid_actions(state)
                if valid_actions:
                    # X plays greedily (epsilon=0) if Q-table exists, otherwise random
                    if Q:
                        action_x = choose_action(
                            state, valid_actions, Q, epsilon=0.0)
                    else:
                        # Random if no training
                        action_x = random.choice(valid_actions)

                    pygame.time.delay(300)  # Small delay for visibility
                    state = apply_action(state, action_x, 'X')
                    print_debug_info(state, Q)

                    # Check game over
                    result = check_winner(state)
                    if result:
                        game_over = True
                        winner = result
                        if result == 'X':
                            message = "X (AI) WINS!"
                        elif result == 'draw':
                            message = "DRAW!"
                    else:
                        message = "O's turn (Human)"

        # Draw header
        draw_header(screen, message, font_medium)

        # Draw board
        draw_board(screen, state, Q, font_large, mode)

        # Draw debug panel (only in AI mode)
        if mode == 'human_vs_ai':
            draw_debug_panel(screen, state, Q, mode,
                             episode_count, font_small, font_medium)

        # Draw controls
        draw_controls(screen, font_small, game_over, mode)

        pygame.display.flip()
        clock.tick(30)


def get_current_turn(state: str) -> str:
    """
    Determine whose turn it is based on board state.
    X always starts, so if equal number of X and O, it's X's turn.
    """
    x_count = state.count('X')
    o_count = state.count('O')
    return 'X' if x_count == o_count else 'O'


def draw_header(screen, message: str, font_medium):
    """Draw the header with title and status message."""
    # Title
    title_font = pygame.font.Font(None, 42)
    title = title_font.render("Tic-Tac-Toe Q-Learning", True, HEADER_COLOR)
    screen.blit(title, (BOARD_OFFSET_X, 20))

    # Message
    msg_surface = font_medium.render(message, True, TEXT_COLOR)
    screen.blit(msg_surface, (BOARD_OFFSET_X, 60))


def draw_board(screen, state: str, Q: Dict, font_large, mode: str = 'human_vs_ai'):
    """Draw the 3x3 Tic-Tac-Toe board."""
    # Draw board background
    board_rect = pygame.Rect(
        BOARD_OFFSET_X, BOARD_OFFSET_Y, CELL_SIZE * 3, CELL_SIZE * 3)
    pygame.draw.rect(screen, (255, 255, 255), board_rect)
    pygame.draw.rect(screen, GRID_COLOR, board_rect, 4)

    # Draw grid lines
    for i in range(1, 3):
        # Vertical lines
        x = BOARD_OFFSET_X + i * CELL_SIZE
        pygame.draw.line(screen, GRID_COLOR,
                         (x, BOARD_OFFSET_Y),
                         (x, BOARD_OFFSET_Y + CELL_SIZE * 3), 3)
        # Horizontal lines
        y = BOARD_OFFSET_Y + i * CELL_SIZE
        pygame.draw.line(screen, GRID_COLOR,
                         (BOARD_OFFSET_X, y),
                         (BOARD_OFFSET_X + CELL_SIZE * 3, y), 3)

    # Highlight best action for X (only in AI mode with trained Q-table)
    current_turn = get_current_turn(state)
    if current_turn == 'X' and mode == 'human_vs_ai' and Q:
        valid_actions = get_valid_actions(state)
        if valid_actions:
            best = get_best_action(state, valid_actions, Q)
            if best >= 0:
                row = best // 3
                col = best % 3
                x = BOARD_OFFSET_X + col * CELL_SIZE
                y = BOARD_OFFSET_Y + row * CELL_SIZE
                # Draw semi-transparent yellow rectangle
                highlight_surface = pygame.Surface(
                    (CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                highlight_surface.fill(HIGHLIGHT_COLOR)
                screen.blit(highlight_surface, (x, y))

    # Draw X's and O's
    for i in range(9):
        row = i // 3
        col = i % 3
        x_center = BOARD_OFFSET_X + col * CELL_SIZE + CELL_SIZE // 2
        y_center = BOARD_OFFSET_Y + row * CELL_SIZE + CELL_SIZE // 2

        if state[i] == 'X':
            # Draw X
            offset = 40
            pygame.draw.line(screen, X_COLOR,
                             (x_center - offset, y_center - offset),
                             (x_center + offset, y_center + offset), 8)
            pygame.draw.line(screen, X_COLOR,
                             (x_center + offset, y_center - offset),
                             (x_center - offset, y_center + offset), 8)
        elif state[i] == 'O':
            # Draw O
            pygame.draw.circle(screen, O_COLOR, (x_center, y_center), 40, 8)


def draw_debug_panel(screen, state: str, Q: Dict, mode: str,
                     episode_count: int, font_small, font_medium):
    """Draw the debug panel on the right side."""
    panel_x = BOARD_OFFSET_X + CELL_SIZE * 3 + 40
    panel_y = 100
    line_height = 24

    # Draw panel background
    panel_width = 360
    panel_height = 520
    panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
    pygame.draw.rect(screen, PANEL_BG, panel_rect)
    pygame.draw.rect(screen, PANEL_BORDER, panel_rect, 2)

    panel_x += 15
    panel_y += 15

    # Title
    title = font_medium.render("DEBUG PANEL", True, HEADER_COLOR)
    screen.blit(title, (panel_x, panel_y))
    panel_y += 35

    # State as a grid
    state_label = font_small.render("Current State:", True, TEXT_COLOR)
    screen.blit(state_label, (panel_x, panel_y))
    panel_y += line_height + 5

    # Draw mini grid representation
    mini_cell = 30
    for i in range(9):
        row = i // 3
        col = i % 3
        x = panel_x + col * (mini_cell + 2) + 10
        y = panel_y + row * (mini_cell + 2)

        # Cell background
        cell_rect = pygame.Rect(x, y, mini_cell, mini_cell)
        pygame.draw.rect(screen, (245, 245, 245), cell_rect)
        pygame.draw.rect(screen, GRID_COLOR, cell_rect, 1)

        # Symbol
        symbol = state[i]
        if symbol != '.':
            color = X_COLOR if symbol == 'X' else O_COLOR
            sym_font = pygame.font.Font(None, 28)
            sym_surface = sym_font.render(symbol, True, color)
            sym_rect = sym_surface.get_rect(
                center=(x + mini_cell//2, y + mini_cell//2))
            screen.blit(sym_surface, sym_rect)

    panel_y += (mini_cell + 2) * 3 + 10

    # Valid actions
    valid_actions = get_valid_actions(state)
    valid_label = font_small.render("Valid Actions:", True, TEXT_COLOR)
    screen.blit(valid_label, (panel_x, panel_y))
    panel_y += line_height

    valid_str = ", ".join(map(str, valid_actions)
                          ) if valid_actions else "None (terminal)"
    valid_text = font_small.render(f"  [{valid_str}]", True, (100, 100, 100))
    screen.blit(valid_text, (panel_x, panel_y))
    panel_y += line_height + 10

    # Q-values
    if valid_actions:
        q_sorted = get_q_values_sorted(state, valid_actions, Q)
        q_title = font_small.render("Q-values (sorted):", True, TEXT_COLOR)
        screen.blit(q_title, (panel_x, panel_y))
        panel_y += line_height

        for a, q in q_sorted[:7]:  # Show top 7
            # Color code based on Q-value
            if q > 0.7:
                color = (39, 174, 96)  # Green for high values
            elif q > 0:
                color = (52, 152, 219)  # Blue for positive
            elif q < 0:
                color = (231, 76, 60)  # Red for negative
            else:
                color = (127, 140, 141)  # Gray for zero

            q_text = font_small.render(f"  action {a} → {q:.4f}", True, color)
            screen.blit(q_text, (panel_x, panel_y))
            panel_y += line_height

        # Best action
        panel_y += 5
        best = get_best_action(state, valid_actions, Q)
        best_label = font_small.render(f"Best Action:", True, TEXT_COLOR)
        screen.blit(best_label, (panel_x, panel_y))
        panel_y += line_height
        best_text = font_small.render(
            f"  {best} (highlighted in yellow)", True, (39, 174, 96))
        screen.blit(best_text, (panel_x, panel_y))
        panel_y += line_height
    else:
        no_actions = font_small.render(
            "Game ended (terminal state)", True, (127, 140, 141))
        screen.blit(no_actions, (panel_x, panel_y))
        panel_y += line_height

    # Divider
    panel_y += 10
    pygame.draw.line(screen, PANEL_BORDER, (panel_x, panel_y),
                     (panel_x + 320, panel_y), 1)
    panel_y += 15

    # Info section
    info_font = pygame.font.Font(None, 22)

    # Mode info
    mode_label = "Human vs AI" if mode == 'human_vs_ai' else mode.replace(
        '_', ' ').title()
    mode_text = info_font.render(f"Mode: {mode_label}", True, TEXT_COLOR)
    screen.blit(mode_text, (panel_x, panel_y))
    panel_y += line_height

    # Q-table size
    size_text = info_font.render(f"Q-table size: {len(Q):,}", True, TEXT_COLOR)
    screen.blit(size_text, (panel_x, panel_y))
    panel_y += line_height

    # Episodes trained
    ep_text = info_font.render(
        f"Episodes trained: ~{episode_count:,}", True, TEXT_COLOR)
    screen.blit(ep_text, (panel_x, panel_y))


def draw_controls(screen, font_small, game_over: bool, mode: str = 'human_vs_ai'):
    """Draw control instructions at the bottom."""
    controls_y = WINDOW_HEIGHT - 90
    controls_x = BOARD_OFFSET_X

    # Background panel for controls
    panel_width = CELL_SIZE * 3
    panel_height = 75
    panel_rect = pygame.Rect(controls_x, controls_y, panel_width, panel_height)
    pygame.draw.rect(screen, PANEL_BG, panel_rect)
    pygame.draw.rect(screen, PANEL_BORDER, panel_rect, 2)

    # Title
    controls_x += 10
    controls_y += 10
    title_font = pygame.font.Font(None, 24)
    title = title_font.render("Controls:", True, HEADER_COLOR)
    screen.blit(title, (controls_x, controls_y))
    controls_y += 22

    # Control list varies by mode
    if mode == 'human_vs_ai':
        controls = [
            "R: Reset  |  S: Save  |  T: Train +5000  |  ESC: Menu"
        ]
    else:  # human_vs_human
        controls = [
            "R: Reset  |  ESC: Menu  |  Click to place marks"
        ]

    if game_over:
        extra_font = pygame.font.Font(None, 20)
        extra = extra_font.render(
            "Press R to play again!", True, (231, 76, 60))
        screen.blit(extra, (controls_x, controls_y))
        controls_y += 20

    for line in controls:
        text = font_small.render(line, True, TEXT_COLOR)
        screen.blit(text, (controls_x, controls_y))
        controls_y += 22


# ============================================================================
# MENU SYSTEM
# ============================================================================

def show_menu(screen) -> str:
    """
    Show main menu and return user's choice.
    Returns: 'human_vs_human', 'human_vs_ai', 'train_ai', or 'quit'
    """
    clock = pygame.time.Clock()
    font_title = pygame.font.Font(None, 56)
    font_button = pygame.font.Font(None, 36)
    font_small = pygame.font.Font(None, 24)

    # Button configuration
    button_width = 400
    button_height = 60
    button_x = (WINDOW_WIDTH - button_width) // 2
    start_y = 200
    button_spacing = 80

    buttons = [
        {'label': '1. Human vs Human', 'value': 'human_vs_human', 'y': start_y},
        {'label': '2. Human vs AI', 'value': 'human_vs_ai',
            'y': start_y + button_spacing},
        {'label': '3. Train AI (Visualized)', 'value': 'train_ai',
         'y': start_y + button_spacing * 2},
        {'label': '4. Quit', 'value': 'quit', 'y': start_y + button_spacing * 3},
    ]

    hovered = None

    while True:
        screen.fill(BG_COLOR)
        mouse_pos = pygame.mouse.get_pos()

        # Title
        title = font_title.render("Tic-Tac-Toe Q-Learning", True, HEADER_COLOR)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 100))
        screen.blit(title, title_rect)

        # Subtitle
        subtitle = font_small.render(
            "Visualize Q-Learning in Action", True, TEXT_COLOR)
        subtitle_rect = subtitle.get_rect(center=(WINDOW_WIDTH // 2, 150))
        screen.blit(subtitle, subtitle_rect)

        # Draw buttons
        hovered = None
        for btn in buttons:
            button_rect = pygame.Rect(
                button_x, btn['y'], button_width, button_height)
            is_hovered = button_rect.collidepoint(mouse_pos)

            if is_hovered:
                hovered = btn['value']
                color = HEADER_COLOR
                border_width = 3
            else:
                color = PANEL_BG
                border_width = 2

            # Draw button
            pygame.draw.rect(
                screen, color if is_hovered else PANEL_BG, button_rect)
            pygame.draw.rect(screen, HEADER_COLOR, button_rect, border_width)

            # Draw text
            text_color = (255, 255, 255) if is_hovered else TEXT_COLOR
            text = font_button.render(btn['label'], True, text_color)
            text_rect = text.get_rect(center=button_rect.center)
            screen.blit(text, text_rect)

        # Instructions
        instructions = font_small.render(
            "Click to select or press 1-4", True, (127, 140, 141))
        instructions_rect = instructions.get_rect(
            center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 50))
        screen.blit(instructions, instructions_rect)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 'quit'
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if hovered:
                    return hovered
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    return 'human_vs_human'
                elif event.key == pygame.K_2:
                    return 'human_vs_ai'
                elif event.key == pygame.K_3:
                    return 'train_ai'
                elif event.key == pygame.K_4 or event.key == pygame.K_ESCAPE:
                    return 'quit'

        pygame.display.flip()
        clock.tick(30)


def draw_training_visualization(screen, fonts, Q, episode, total_episodes, epsilon,
                                episode_history, final_state):
    """
    Visualize the training process - show board state, Q-values, rewards, etc.
    """
    screen.fill(BG_COLOR)
    font_large, font_medium, font_small = fonts

    # Title
    title = font_medium.render(
        f"Training Episode {episode + 1} / {total_episodes}", True, HEADER_COLOR)
    screen.blit(title, (50, 20))

    # Epsilon and Q-table info
    info_text = font_small.render(
        f"Epsilon: {epsilon:.4f}  |  Q-table size: {len(Q):,}  |  Press ESC to stop", True, TEXT_COLOR)
    screen.blit(info_text, (50, 55))

    # Show final state of the episode
    board_x = 100
    board_y = 150
    mini_cell = 100

    # Draw board
    for i in range(9):
        row = i // 3
        col = i % 3
        x = board_x + col * (mini_cell + 3)
        y = board_y + row * (mini_cell + 3)

        # Cell
        cell_rect = pygame.Rect(x, y, mini_cell, mini_cell)
        pygame.draw.rect(screen, (255, 255, 255), cell_rect)
        pygame.draw.rect(screen, GRID_COLOR, cell_rect, 3)

        # Symbol
        symbol = final_state[i]
        if symbol != '.':
            color = X_COLOR if symbol == 'X' else O_COLOR
            sym_font = pygame.font.Font(None, 72)
            sym_surface = sym_font.render(symbol, True, color)
            sym_rect = sym_surface.get_rect(
                center=(x + mini_cell//2, y + mini_cell//2))
            screen.blit(sym_surface, sym_rect)

    # Episode history panel
    panel_x = 450
    panel_y = 100
    panel_width = 500
    panel_height = 480

    panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
    pygame.draw.rect(screen, PANEL_BG, panel_rect)
    pygame.draw.rect(screen, PANEL_BORDER, panel_rect, 2)

    # Panel content
    panel_x += 15
    panel_y += 15
    line_height = 22

    header = font_medium.render("Episode Summary", True, HEADER_COLOR)
    screen.blit(header, (panel_x, panel_y))
    panel_y += 35

    # Show moves
    if episode_history:
        moves_text = font_small.render(
            f"Total moves: {len(episode_history)}", True, TEXT_COLOR)
        screen.blit(moves_text, (panel_x, panel_y))
        panel_y += line_height + 10

        # Show last few moves with rewards
        for idx, (state_before, action, reward, state_after, status) in enumerate(episode_history[-6:]):
            move_num = len(episode_history) - 6 + idx + \
                1 if len(episode_history) > 6 else idx + 1

            # Color code reward
            if reward > 0:
                reward_color = (39, 174, 96)  # Green
            elif reward < 0:
                reward_color = (231, 76, 60)  # Red
            else:
                reward_color = (127, 140, 141)  # Gray

            move_text = font_small.render(
                f"Move {move_num}: action={action}, reward={reward:.2f}", True, TEXT_COLOR)
            screen.blit(move_text, (panel_x, panel_y))
            panel_y += line_height

            status_text = font_small.render(
                f"  Status: {status}", True, reward_color)
            screen.blit(status_text, (panel_x, panel_y))
            panel_y += line_height + 5

    # Q-learning formula visualization
    panel_y = panel_rect.bottom - 100
    formula_title = font_small.render("Q-Learning Update:", True, HEADER_COLOR)
    screen.blit(formula_title, (panel_x, panel_y))
    panel_y += line_height

    formula_text = font_small.render(
        "Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]", True, TEXT_COLOR)
    screen.blit(formula_text, (panel_x, panel_y))
    panel_y += line_height

    params = font_small.render(f"α={ALPHA}, γ={GAMMA}", True, (100, 100, 100))
    screen.blit(params, (panel_x, panel_y))

    pygame.display.flip()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main entry point with menu system.
    1. Show menu
    2. Handle user's choice (human vs human, human vs AI, or train AI)
    """
    print("=" * 60)
    print("TIC-TAC-TOE Q-LEARNING")
    print("=" * 60)

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Tic-Tac-Toe Q-Learning")

    # Try to load existing Q-table
    Q = load_qtable(Q_TABLE_FILE)
    if Q:
        print(f"\nQ-table loaded: {len(Q)} entries")
    else:
        print("\nNo saved Q-table found. You can train one from the menu.")
        Q = {}

    while True:
        # Show menu
        choice = show_menu(screen)

        if choice == 'quit':
            break

        elif choice == 'human_vs_human':
            print("\n--- Human vs Human Mode ---")
            run_game(Q, mode='human_vs_human')

        elif choice == 'human_vs_ai':
            if not Q:
                print("\n*** Warning: No trained Q-table! AI will play randomly. ***")
                print("Consider training first for better gameplay.")
            print("\n--- Human vs AI Mode ---")
            run_game(Q, mode='human_vs_ai')

        elif choice == 'train_ai':
            print("\n--- Training Mode (Visualized) ---")
            print(f"Training {EPISODES} episodes...")

            # Prepare fonts for visualization
            fonts = (
                pygame.font.Font(None, 80),   # font_large
                pygame.font.Font(None, 36),   # font_medium
                pygame.font.Font(None, 24)    # font_small
            )

            # Train with visualization
            completed = train(Q, EPISODES, ALPHA, GAMMA, EPSILON_START,
                              EPSILON_MIN, EPSILON_DECAY,
                              visualize=True, screen=screen, fonts=fonts)

            if completed:
                save_qtable(Q, Q_TABLE_FILE)
                print("\nTraining complete and Q-table saved!")

                # Show completion message
                screen.fill(BG_COLOR)
                font_large = pygame.font.Font(None, 48)
                msg1 = font_large.render(
                    "Training Complete!", True, (39, 174, 96))
                msg2 = fonts[1].render(
                    f"Q-table has {len(Q):,} entries", True, TEXT_COLOR)
                msg3 = fonts[2].render(
                    "Click anywhere to continue...", True, (127, 140, 141))

                screen.blit(msg1, (WINDOW_WIDTH//2 -
                            200, WINDOW_HEIGHT//2 - 50))
                screen.blit(msg2, (WINDOW_WIDTH//2 -
                            150, WINDOW_HEIGHT//2 + 10))
                screen.blit(msg3, (WINDOW_WIDTH//2 -
                            140, WINDOW_HEIGHT//2 + 60))
                pygame.display.flip()

                # Wait for click
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            waiting = False
                        elif event.type in [pygame.MOUSEBUTTONDOWN, pygame.KEYDOWN]:
                            waiting = False
            else:
                print("\nTraining stopped by user.")

    pygame.quit()
    print("\nGame closed. Goodbye!")


if __name__ == "__main__":
    main()
