# Chrome T-Rex Runner Bot 🦖

An autonomous Python bot built with [Playwright](https://playwright.dev/python/) that plays the Chrome Dinosaur Game ([wayou.github.io/t-rex-runner](https://wayou.github.io/t-rex-runner/)) in an endless loop.

Unlike simple pixel-scanning scripts, this bot interacts directly with the internal JavaScript game engine (`Runner.instance_`, `tRex`, `horizon.obstacles`), enabling frame-accurate reactions at any speed.

## ✨ Features

* **Direct Engine Integration:** Reads exact coordinates, dimensions, and obstacle types straight from `Runner.instance_`.
* **Handles All Obstacle Types:**
  * **Cacti** (`CACTUS_SMALL` / `CACTUS_LARGE`) → executes a full-height jump.
  * **Low & Mid-Altitude Pterodactyls** (`yPos` 75 / 100) → executes a jump.
  * **High-Altitude Pterodactyls** (`yPos` 50) → holds crouch (`ArrowDown`) until the obstacle safely passes overhead.
* **Dynamic Speed Scaling:** Lead distance automatically scales with `currentSpeed` so the bot maintains accurate timing as the game accelerates.
* **Classic In-Game Score Overlay:** Renders a minimal, retro-styled `HI 00000 00000` counter in the top-right corner matching the original Chrome aesthetic.
* **Endless Loop Mode:** Automatically restarts the run after a crash and tracks the session's highest score.

## 🚀 Quick Start

### 1. Prerequisites
Make sure you have Python 3.8+ installed, then install Playwright and the Chromium browser binary:

```bash
pip install playwright
python -m playwright install chromium
```

### 2. Run the Bot
```bash
python dino_bot-v4.py
```
Press `Ctrl+C` in the terminal at any time to stop the bot.

## ⚙️ Environment Variables

You can customize execution behavior using environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `HEADLESS` | `0` | Set to `1` to run in headless mode (without opening a visible browser window). |
| `CHANNEL` | `""` | Set to `chrome` to use your system-installed Google Chrome instead of bundled Chromium. |
| `RUN_SECONDS` | `0` | Automatically stop the bot after `N` seconds (`0` = run indefinitely). |

**Example (run headless for 60 seconds):**
```bash
HEADLESS=1 RUN_SECONDS=60 python dino_bot-v4.py
```

## 🔧 Configuration Constants

You can fine-tune the bot's timing directly at the top of `dino_bot-v4.py`:

* `STEP_DELAY = 0.025` — Main loop interval in seconds (~1.5 frames at 60 FPS).
* `JUMP_APEX_FRAMES = 15` — Number of frames before jump apex used to calculate jump lead distance.
* `DUCK_LEAD_FRAMES = 9` — Lead time in frames before ducking under high pterodactyls.
* `PRINT_EVERY = 40` — Console status logging frequency (in ticks).

## 📁 Project Structure

```text
.
├── dino_bot-v4.py   # Main bot script with Playwright automation & HUD overlay
└── README.md        # Project documentation
```

## 📄 License

This project is licensed under the MIT License — feel free to use, modify, and distribute it.
