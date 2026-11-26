# Chrome T-Rex Runner Bot 🦖

An autonomous Python bot built with [Playwright](https://playwright.dev/python/) that plays the Chrome Dinosaur Game ([wayou.github.io/t-rex-runner](https://wayou.github.io/t-rex-runner/)) in an endless loop.

Unlike simple pixel-scanning scripts, this bot interacts directly with the internal JavaScript game engine (`Runner.instance_`, `tRex`, `horizon.obstacles`), enabling frame-accurate reactions at any speed. On top of the rule-based engine, the bot maintains a **Q-learning table (3×2)** in parallel, which updates on every step via the Bellman equation — a self-contained demonstration of reinforcement learning.

## ✨ Features

* **Direct Engine Integration:** Reads exact coordinates, dimensions, and obstacle types straight from `Runner.instance_`.
* **Handles All Obstacle Types:**
  * **Cacti** (`CACTUS_SMALL` / `CACTUS_LARGE`) → executes a full-height jump.
  * **Low & Mid-Altitude Pterodactyls** (`yPos` 75 / 100) → executes a jump.
  * **High-Altitude Pterodactyls** (`yPos` 50) → holds crouch (`ArrowDown`) until the obstacle safely passes overhead.
* **Dynamic Speed Scaling:** Lead distance automatically scales with `currentSpeed` so the bot maintains accurate timing as the game accelerates.
* **Parallel Q-Table (3×2):** A reinforcement-learning table runs alongside the rule engine. It is **not** used to make jump decisions — the rules do that — but it is updated every step using the Bellman equation and printed to the terminal at the end of the run as a learning artefact.
* **Classic In-Game Score Overlay:** Renders a minimal, retro-styled `HI 00000 00000` counter in the top-right corner matching the original Chrome aesthetic.
* **Endless Loop Mode:** Automatically restarts the run after a crash and tracks the session's highest score.

## 🧠 Q-Table (Reinforcement Learning)

The bot keeps a Q-table in memory as a plain Python dictionary — no neural network, no files on disk.

**Structure (3 states × 2 actions):**

| State | Meaning               | Action 0      | Action 1  |
| :---: | :-------------------- | :------------ | :-------- |
| 0     | obstacle far (>250px) | don't jump    | jump      |
| 1     | obstacle mid (120–250px) | don't jump | jump      |
| 2     | obstacle close (<120px) | don't jump  | jump      |

Each cell holds **Q(s, a)** — the expected sum of future rewards for taking action *a* in state *s*. On every live step the bot receives `+1`; on a crash it receives `−100`. The cell is updated with the Bellman equation:

```
Q(s,a) ← Q(s,a) + α · [ r + γ · max Q(s',a') − Q(s,a) ]
```

> **Note:** the rule engine still decides when to jump/duck. The Q-table only *records* what happened, so the bot plays reliably from the start while still demonstrating a working RL loop.

## 🚀 Quick Start

### 1. Prerequisites
Make sure you have Python 3.8+ installed, then install Playwright and the Chromium browser binary:

```bash
pip install playwright
python -m playwright install chromium
```

### 2. Run the Bot
```bash
python dino.py
```
Press `Ctrl+C` in the terminal at any time to stop the bot. The final Q-table is then printed to the console.

## ⚙️ Environment Variables

You can customize execution behavior using environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `HEADLESS` | `0` | Set to `1` to run in headless mode (without opening a visible browser window). |
| `CHANNEL` | `""` | Set to `chrome` to use your system-installed Google Chrome instead of bundled Chromium. |
| `RUN_SECONDS` | `0` | Automatically stop the bot after `N` seconds (`0` = run indefinitely). |

**Example (run headless for 60 seconds):**
```bash
HEADLESS=1 RUN_SECONDS=60 python dino.py
```

## 🔧 Configuration Constants

You can fine-tune the bot's timing directly at the top of `dino.py`:

**Rule engine:**
* `STEP_DELAY = 0.025` — Main loop interval in seconds (~1.5 frames at 60 FPS).
* `JUMP_APEX_FRAMES = 15` — Number of frames before jump apex used to calculate jump lead distance.
* `DUCK_LEAD_FRAMES = 9` — Lead time in frames before ducking under high pterodactyls.
* `PRINT_EVERY = 40` — Console status logging frequency (in ticks).

**Q-learning:**
* `STATE_FAR_MAX = 250` — Distance (px) above which an obstacle counts as "far" (state 0).
* `STATE_MID_MAX = 120` — Distance (px) above which an obstacle counts as "mid" (state 1); below it is "close" (state 2).
* `ALPHA = 0.2` — Learning rate (how strongly new experience overwrites old values).
* `GAMMA = 0.9` — Discount factor for future rewards.
* `REWARD_ALIVE = 1` — Reward for surviving one step.
* `REWARD_CRASH = -100` — Penalty for crashing into an obstacle.

## 📁 Project Structure

```text
.
├── dino.py           # Main bot script: rule engine + parallel Q-table
└── README.md         # Project documentation
```

## 📄 License

This project is licensed under the MIT License — feel free to use, modify, and distribute it.
