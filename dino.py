import os
import sys
import time
from playwright.sync_api import sync_playwright

GAME_URL = "https://wayou.github.io/t-rex-runner/"

STEP_DELAY = 0.025
JUMP_APEX_FRAMES = 15
DUCK_LEAD_FRAMES = 9
PRINT_EVERY = 40

STATE_FAR_MAX = 250
STATE_MID_MAX = 120
NUM_STATES = 3
NUM_ACTIONS = 2

ALPHA = 0.2
GAMMA = 0.9
REWARD_ALIVE = 1
REWARD_CRASH = -100

HEADLESS = os.environ.get("HEADLESS", "0") == "1"
CHANNEL = os.environ.get("CHANNEL", "")
RUN_SECONDS = float(os.environ.get("RUN_SECONDS", "0"))


def new_q_table():
    return {
        0: [0.0, 0.0],
        1: [0.0, 0.0],
        2: [0.0, 0.0],
    }


def print_q_table(table):
    print()
    print("Итоговая Q-таблица 3x2:")
    print("  состояние | не прыгать (0) | прыгать (1)")
    print("  ----------+----------------+------------")
    names = {0: "далеко", 1: "средне", 2: "близко"}
    for s in range(NUM_STATES):
        q0, q1 = table[s]
        print(f"  {s} · {names[s]:6} | {q0:>14.2f} | {q1:>11.2f}")
    print()
    print("Выгодное действие по таблице (argmax):")
    for s in range(NUM_STATES):
        best = 0 if table[s][0] >= table[s][1] else 1
        name = "НЕ прыгать" if best == 0 else "прыгать"
        print(f"  состояние {s} ({names[s]:6}) -> {name}")


INJECT_OVERLAY_JS = r"""
() => {
    if (document.getElementById('bot_overlay')) return;
    const box = document.createElement('div');
    box.id = 'bot_overlay';
    box.style.cssText = [
        'position:fixed', 'top:14px', 'right:20px', 'z-index:999999',
        'background:transparent', 'color:#535353',
        'font:700 16px/1 "Courier New",Courier,monospace',
        'pointer-events:none', 'letter-spacing:2px', 'white-space:nowrap'
    ].join(';');
    box.innerHTML =
        '<span style="color:#757575">HI <span id="bot_rec">00000</span></span>' +
        '&nbsp;&nbsp;' +
        '<span style="color:#535353" id="bot_dist">00000</span>';
    document.body.appendChild(box);
}
"""

READ_STATE_JS = r"""
() => {
    const r = window.Runner && window.Runner.instance_;
    const score = r ? Math.round(r.distanceRan * 0.025) : 0;
    const hi    = r ? Math.round(r.highestScore * 0.025) : 0;
    const ov = document.getElementById('bot_overlay');
    if (ov) {
        const dEl = document.getElementById('bot_dist');
        const rEl = document.getElementById('bot_rec');
        if (dEl) dEl.textContent = String(score).padStart(5, '0');
        if (rEl) rEl.textContent = String(hi).padStart(5, '0');
    }
    if (!r) return { status: 'NO_RUNNER', score, highScore: hi };
    if (r.crashed) return { status: 'DEAD', score, highScore: hi };
    if (!r.playing) return { status: 'NOT_PLAYING', score, highScore: hi };
    const d = r.tRex;
    const list = r.horizon.obstacles;
    const o = list.length ? list[0] : null;
    return {
        status: 'OK',
        speed: r.currentSpeed,
        dinoX: d.xPos,
        jumping: d.jumping,
        ducking: d.ducking,
        score,
        highScore: hi,
        obstacle: o ? {
            x: o.xPos,
            y: o.yPos,
            type: o.typeConfig.type,
            w: o.typeConfig.width * o.size
        } : null
    };
}
"""


class DinoBot:

    def __init__(self, page):
        self.page = page
        self.holding_jump = False
        self.holding_duck = False
        self.tick = 0
        self.episode = 1
        self.last_score = 0
        self.last_speed = 0.0
        self.best_score = 0
        self.q_table = new_q_table()
        self.last_state = None
        self.last_action = None

    def _press_jump(self):
        self.page.keyboard.press("Space")

    def _duck_down(self):
        if not self.holding_duck:
            self.page.keyboard.down("ArrowDown")
            self.holding_duck = True

    def _duck_up(self):
        if self.holding_duck:
            self.page.keyboard.up("ArrowDown")
            self.holding_duck = False

    def _release_all(self):
        if self.holding_jump:
            self.page.keyboard.up("Space")
            self.holding_jump = False
        self._duck_up()

    def _restart(self):
        self._release_all()
        self.page.keyboard.press("Space")
        time.sleep(0.4)
        self.episode += 1
        self.last_state = None
        self.last_action = None

    def _state_of(self, obstacle, dino_x):
        if not obstacle:
            return 0
        dist = obstacle["x"] - dino_x
        if dist > STATE_FAR_MAX:
            return 0
        if dist > STATE_MID_MAX:
            return 1
        return 2

    def _update_q(self, s, a, reward, s_next):
        old = self.q_table[s][a]
        best_next = max(self.q_table[s_next])
        self.q_table[s][a] = old + ALPHA * (reward + GAMMA * best_next - old)

    def step(self):
        state = self.page.evaluate(READ_STATE_JS)
        status = state.get("status")
        self.best_score = state.get("highScore", 0)

        if status == "NO_RUNNER":
            return
        if status == "DEAD":
            if self.last_state is not None:
                self._update_q(self.last_state, self.last_action,
                               REWARD_CRASH, 0)
            print(f"[эпизод {self.episode}] крах. пробежал={state.get('score', 0)}, "
                  f"рекорд={self.best_score}")
            self._restart()
            return
        if status == "NOT_PLAYING":
            self.page.keyboard.press("Space")
            time.sleep(0.2)
            return
        if status != "OK":
            return

        self.last_score = state.get("score", 0)
        self.last_speed = speed = state["speed"]
        obstacle = state.get("obstacle")

        if self.holding_jump and not state["jumping"]:
            self.page.keyboard.up("Space")
            self.holding_jump = False

        if not obstacle:
            self._duck_up()
            return

        obs = obstacle
        dist = obs["x"] - state["dinoX"]
        obs_w = obs["w"]

        is_high_ptero = (obs["type"] == "PTERODACTYL" and obs["y"] <= 50)

        jumped_now = False
        if is_high_ptero:
            self._handle_high_ptero(dist, obs_w, speed)
            action = 0
        else:
            jumped_now = self._handle_jump_obstacle(dist, obs_w, speed, state)
            action = 1 if jumped_now else 0

        s = self._state_of(obs, state["dinoX"])
        if self.last_state is not None:
            self._update_q(self.last_state, self.last_action,
                           REWARD_ALIVE, s)
        self.last_state = s
        self.last_action = action

    def _handle_high_ptero(self, dist, obs_w, speed):
        duck_start = max(55.0, speed * DUCK_LEAD_FRAMES)
        if duck_start >= dist > -obs_w:
            self._duck_down()
        if self.holding_duck and dist <= -obs_w:
            self._duck_up()

    def _handle_jump_obstacle(self, dist, obs_w, speed, state):
        self._duck_up()
        if state["jumping"] or self.holding_jump:
            return False
        center_dist = (obs_w / 2) + dist - 22
        reaction_px = speed * JUMP_APEX_FRAMES
        if 0 < center_dist <= reaction_px:
            self._press_jump()
            return True
        return False


def main():
    print(f"Запуск бота. headless={HEADLESS}, channel={CHANNEL or 'chromium (bundled)'}")
    launch_kwargs = {"headless": HEADLESS}
    if CHANNEL:
        launch_kwargs["channel"] = CHANNEL
    launch_kwargs["args"] = ["--disable-gpu", "--no-sandbox"]

    with sync_playwright() as p:
        browser = p.chromium.launch(**launch_kwargs)
        page = browser.new_page(viewport={"width": 800, "height": 200})
        print("Открываем игру...")
        page.goto(GAME_URL, wait_until="domcontentloaded")
        time.sleep(2.0)
        page.evaluate(INJECT_OVERLAY_JS)

        try:
            page.click("canvas", timeout=3000)
        except Exception:
            pass
        page.keyboard.press("Space")
        time.sleep(0.6)

        bot = DinoBot(page)
        print("Бот начал игру. Ctrl+C — выход.")
        print("Q-таблица 3x2 ведётся параллельно в памяти программы")
        print("и будет напечатана в терминале по завершении работы.")
        print()

        start = time.time()
        try:
            while True:
                bot.step()
                bot.tick += 1
                if bot.tick % PRINT_EVERY == 0:
                    elapsed = time.time() - start
                    print(f"[{elapsed:6.1f}s] пробежал {bot.last_score:>5} | "
                          f"рекорд {bot.best_score:>5} | "
                          f"скорость {bot.last_speed:.1f} | "
                          f"эпизод {bot.episode}")
                time.sleep(STEP_DELAY)
                if RUN_SECONDS and (time.time() - start) >= RUN_SECONDS:
                    print()
                    print(f"Тест завершён: {RUN_SECONDS}s. "
                          f"рекорд={bot.best_score}, эпизодов={bot.episode}")
                    break
        except KeyboardInterrupt:
            print()
            print("Остановлено пользователем.")
        finally:
            print_q_table(bot.q_table)
            browser.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
