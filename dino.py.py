"""
Улучшенный бот для игры https://wayou.github.io/t-rex-runner/

Что улучшено по сравнению с исходным Q-learning ботом:
  • Бот работает с реальным API игры (Runner.instance_, tRex, horizon.obstacles),
    а не с абстрактными «состояниями по дистанции».
  • Обрабатываются ВСЕ типы препятствий:
      - кактусы (CACTUS_SMALL / CACTUS_LARGE)        → прыжок
      - птеродактиль на низкой/средней высоте (yPos 75/100) → прыжок
      - птеродактиль на уровне головы (yPos 50)      → приседание (удержание)
  • Дистанция реакции масштабируется текущей скоростью игры
    (currentSpeed), чтобы бот не врезался на высоких скоростях.
  • Прыжок делается через нажатие Space (быстрый тап = полный прыжок
    до MAX_JUMP_HEIGHT — хватает для любого кактуса).
  • Приседание удерживается, пока птеродактиль не пролетит мимо.
  • Бесконечный режим: после краха игра автоматически перезапускается.

Установка (один раз):
    pip install playwright
    python -m playwright install chromium
Запуск:
    python dino_bot.py
Переменные окружения (для тестов):
    HEADLESS=1      — без окна браузера
    CHANNEL=chrome  — использовать системный Google Chrome вместо bundled Chromium
    RUN_SECONDS=30  — проиграть N секунд и выйти (иначе — бесконечно)
"""

import os
import sys
import time
from playwright.sync_api import sync_playwright

GAME_URL = "https://wayou.github.io/t-rex-runner/"

# --- Настройки бота ---------------------------------------------------------
STEP_DELAY = 0.025          # пауза между тиками (с) — ~1.5 кадра @60fps
JUMP_APEX_FRAMES = 15       # кадров до верхней точки прыжка (для упреждения)
DUCK_LEAD_FRAMES = 9        # упреждение для приседания (кадры)
PRINT_EVERY = 40            # каждые N тиков печатать статус

HEADLESS = os.environ.get("HEADLESS", "0") == "1"
# По умолчанию — bundled Chromium (ставится командой `playwright install chromium`).
# Если хотите использовать системный Google Chrome: CHANNEL=chrome
CHANNEL = os.environ.get("CHANNEL", "")
RUN_SECONDS = float(os.environ.get("RUN_SECONDS", "0"))  # 0 → бесконечно


# --- Оверлей на экране: родной шрифт и вид как в самой игре -----------------
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

# --- Чтение состояния игры + обновление оверлея (один evaluate-вызов) ------
# Счёт и рекорд берутся САМОЙ ИГРОЙ: distanceRan × 0.025 (COEFFICIENT из
# DistanceMeter.config) — это ровно то, что игра рисует на канвасе.
# Рекорд = Runner.highestScore × 0.025 — это родной «HI» игры.
READ_STATE_JS = r"""
() => {
    const r = window.Runner && window.Runner.instance_;
    const score = r ? Math.round(r.distanceRan * 0.025) : 0;
    const hi    = r ? Math.round(r.highestScore * 0.025) : 0;
    // Обновляем оверлей на экране, если он есть.
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
            type: o.typeConfig.type,        // 'CACTUS_SMALL' | 'CACTUS_LARGE' | 'PTERODACTYL'
            w: o.typeConfig.width * o.size
        } : null
    };
}
"""


class DinoBot:
    """Бот, который играет в T-Rex Runner бесконечно."""

    def __init__(self, page):
        self.page = page
        self.holding_jump = False
        self.holding_duck = False
        self.tick = 0
        self.episode = 1
        self.last_score = 0
        self.last_speed = 0.0
        self.best_score = 0
        # Рекорд берётся у самой игры (Runner.highestScore) — см. step().

    # -- низкоуровневые действия -------------------------------------------
    def _press_jump(self):
        # Быстрый тап Space → полный прыжок (endJump не срабатывает раньше
        # минимальной высоты, поэтому динозаврик допрыгивает до MAX_JUMP_HEIGHT).
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

    # -- перезапуск ---------------------------------------------------------
    def _restart(self):
        self._release_all()
        self.page.keyboard.press("Space")
        time.sleep(0.4)
        self.episode += 1

    # -- главный цикл -------------------------------------------------------
    def step(self):
        state = self.page.evaluate(READ_STATE_JS)
        status = state.get("status")
        # Рекорд — родной «HI» игры (Runner.highestScore × 0.025).
        self.best_score = state.get("highScore", 0)

        if status == "NO_RUNNER":
            return
        if status == "DEAD":
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

        # Сброс удержания прыжка после приземления
        if self.holding_jump and not state["jumping"]:
            self.page.keyboard.up("Space")
            self.holding_jump = False

        if not obstacle:
            # Препятствия нет — можно встать (если присели без причины)
            self._duck_up()
            return

        obs = obstacle
        dist = obs["x"] - state["dinoX"]      # от динозавра до левой кромки препятствия
        obs_w = obs["w"]

        # Высокий птеродактиль (yPos == 50) → приседать; всё остальное → прыгать
        is_high_ptero = (obs["type"] == "PTERODACTYL" and obs["y"] <= 50)

        if is_high_ptero:
            self._handle_high_ptero(dist, obs_w, speed)
        else:
            self._handle_jump_obstacle(dist, obs_w, speed, state)

    # -- обработка высокого птеродактиля -----------------------------------
    def _handle_high_ptero(self, dist, obs_w, speed):
        duck_start = max(55.0, speed * DUCK_LEAD_FRAMES)
        # Пока птеродактиль близко — приседаем и держим
        if duck_start >= dist > -obs_w:
            self._duck_down()
        # Пролетел мимо (правая кромка за динозавриком) — встать
        if self.holding_duck and dist <= -obs_w:
            self._duck_up()

    # -- обработка кактуса / низкого птеродактиля --------------------------
    def _handle_jump_obstacle(self, dist, obs_w, speed, state):
        # Перед прыжком обязательно встать (игра не прыгает из приседа)
        self._duck_up()
        if state["jumping"] or self.holding_jump:
            return
        # Упреждение в пикселях = скорость × время до апекса.
        # Прыжок стартует так, чтобы в верхней точке динозаврик оказался
        # над центром препятствия — это даёт максимальный запас по высоте.
        center_dist = (obs_w / 2) + dist - 22  # 22 ≈ половина ширины динозавра
        reaction_px = speed * JUMP_APEX_FRAMES
        if 0 < center_dist <= reaction_px:
            self._press_jump()


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
        # Внедряем оверлей в стиле самой игры (HI 00000  00000).
        page.evaluate(INJECT_OVERLAY_JS)

        # Фокус на канвас и старт
        try:
            page.click("canvas", timeout=3000)
        except Exception:
            pass
        page.keyboard.press("Space")
        time.sleep(0.6)

        bot = DinoBot(page)
        print("Бот начал игру. Ctrl+C — выход.")
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
            browser.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
