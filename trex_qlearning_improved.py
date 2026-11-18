import os
import time
import random

import numpy as np
from playwright.sync_api import sync_playwright


GAME_URL = "https://wayou.github.io/t-rex-runner/"

# ===== Параметры Q-обучения =====
ALPHA = 0.2            # скорость обучения
GAMMA = 0.9            # дисконт
EPSILON_START = 0.15   # стартовая случайность ( exploration )
EPSILON_MIN = 0.02     # нижняя граница exploration
EPSILON_DECAY = 0.99   # затухание exploration после каждой игры
STEP_DELAY = 0.03      # задержка между шагами, секунды

MODEL_FILE = "q_table_inf.npy"
EPSILON_FILE = "q_table_inf.epsilon"

TRAIN_MODE = True
LOAD_MODEL = True
SAVE_MODEL = True
SAVE_EVERY = 5         # сохранять Q-таблицу каждые N игр
RESET_MODEL = False    # True = игнорировать файл, начать с приоров (чистый старт)
DEBUG_STEPS = True     # логировать каждую смену состояния / действие

# ===== Действия =====
ACTION_RUN = 0         # просто бежать (отпустить присед, не прыгать)
ACTION_JUMP = 1        # прыжок (Space)
ACTION_DUCK = 2        # присесть (удерживать ArrowDown)
ACTION_NAMES = {0: "БЕЖАТЬ", 1: "ПРЫГАТЬ", 2: "ПРИСЕДАТЬ"}
NUM_ACTIONS = 3

# ===== Пространство состояний (9 дискретных состояний) =====
#   0 - препятствия нет (или высоко летящая птица - можно просто бежать)
#   1 - далеко        + прыгать   (ttc > 50 кадров)
#   2 - далеко        + приседать (ttc > 50 кадров)
#   3 - средне        + прыгать   (25 < ttc <= 50)
#   4 - средне        + приседать (25 < ttc <= 50)
#   5 - близко РАНО   + прыгать   (15 < ttc <= 25) - ОПТИМАЛЬНО прыгать
#   6 - близко РАНО   + приседать (15 < ttc <= 25) - пора приседать
#   7 - близко ПОЗДНО + прыгать   (ttc <= 15) - последний шанс / уже поздно
#   8 - близко ПОЗДНО + приседать (ttc <= 15) - присед мгновенный, ещё ок
#
#   TTC в КАДРАХ = distance / currentSpeed (speed = пиксели/кадр при ~60 FPS,
#   а НЕ пиксели/сек!). Прыжок длится ~43 кадра, пик ~22 -> прыгать при ttc ~18-25.
NUM_STATES = 9

# ===== Приоры (готовая начальная стратегия) =====
# Бот играет сносно с ПЕРВОГО запуска. Q-обучение уточняет.
# БЕЖАТЬ пока далеко/средне; ПРЫГАТЬ/ПРИСЕДАТЬ в "близко рано".
# В "близко поздно" для прыжка - слабый приор (последний шанс);
# для приседания приор сохраняется (присед мгновенный).
# БЕЖАТЬ=0 в действующих состояниях, чтобы в воздухе бот выбирал БЕЖАТЬ
# (дать прыжку завершиться), а не ПРИСЕДАТЬ.
PRIORS = {
    0: [ 5.0, -5.0, -5.0],  # нет препятствия           -> БЕЖАТЬ
    1: [ 5.0, -5.0, -5.0],  # далеко + прыгать          -> БЕЖАТЬ
    2: [ 5.0, -5.0, -5.0],  # далеко + приседать        -> БЕЖАТЬ
    3: [ 5.0, -5.0, -5.0],  # средне + прыгать          -> БЕЖАТЬ
    4: [ 5.0, -5.0, -5.0],  # средне + приседать        -> БЕЖАТЬ
    5: [ 0.0,  5.0, -5.0],  # близко рано + прыгать     -> ПРЫГАТЬ
    6: [ 0.0, -5.0,  5.0],  # близко рано + приседать   -> ПРИСЕДАТЬ
    7: [ 0.0,  2.0, -5.0],  # близко поздно + прыгать   -> ПРЫГАТЬ (последний шанс)
    8: [ 0.0, -5.0,  5.0],  # близко поздно + приседать -> ПРИСЕДАТЬ
}

# Короткие теги для построчного лога шагов
STATE_TAGS = {
    0: "свободно",       1: "далеко+прыг",    2: "далеко+присед",
    3: "средне+прыг",    4: "средне+присед",
    5: "БЛИЗКО_РАНО+прыг", 6: "БЛИЗКО_РАНО+присед",
    7: "БЛИЗКО_ПОЗДНО+прыг", 8: "БЛИЗКО_ПОЗДНО+присед",
}

q_table = {s: list(PRIORS[s]) for s in range(NUM_STATES)}

epsilon = EPSILON_START
duck_held = False  # удерживается ли ArrowDown в данный момент


def load_q_table():
    """
    Загружает Q-таблицу. Если файла нет, он несовместим (старая схема состояний
    после исправления TTC-бага) или RESET_MODEL=True - инициализирует приорами,
    чтобы бот играл сносно сразу.
    """
    global q_table
    use_priors = True

    if LOAD_MODEL and not RESET_MODEL and os.path.exists(MODEL_FILE):
        try:
            loaded = np.load(MODEL_FILE, allow_pickle=True).item()
        except Exception:
            loaded = {}

        # Формат валиден, только если есть все состояния по 3 значения.
        is_valid = (
            isinstance(loaded, dict)
            and all(s in loaded for s in range(NUM_STATES))
            and all(len(loaded[s]) == NUM_ACTIONS for s in range(NUM_STATES))
        )

        if is_valid:
            for s in range(NUM_STATES):
                q_table[s] = [float(x) for x in loaded[s]]
            use_priors = False
            print(f"Загружена сохранённая Q-таблица ({NUM_STATES} состояний, 3 действия).")

    if use_priors:
        for s in range(NUM_STATES):
            q_table[s] = list(PRIORS[s])
        if RESET_MODEL:
            print("Принудительный сброс: Q-таблица инициализирована приорами.")
        elif LOAD_MODEL and os.path.exists(MODEL_FILE):
            print("Старая Q-таблица несовместима с новой схемой состояний. "
                  "Инициализирована приорами (готовая стратегия).")
        else:
            print("Q-таблица создана с нуля (инициализирована приорами).")


def load_epsilon():
    global epsilon
    if RESET_MODEL:
        epsilon = EPSILON_START
        print(f"Сброс epsilon={epsilon:.3f} (RESET_MODEL)")
        return
    if LOAD_MODEL and os.path.exists(EPSILON_FILE):
        try:
            with open(EPSILON_FILE) as f:
                # держим затухавшее значение, ограничиваем только снизу
                epsilon = max(EPSILON_MIN, float(f.read().strip()))
            print(f"Загружен epsilon={epsilon:.3f}")
        except Exception:
            epsilon = EPSILON_START


def save_q_table():
    if SAVE_MODEL:
        np.save(MODEL_FILE, q_table)
        print(f"Q-таблица сохранена в файл {MODEL_FILE}")


def save_epsilon():
    if SAVE_MODEL:
        with open(EPSILON_FILE, "w") as f:
            f.write(str(epsilon))


def get_state(page):
    """
    Читает состояние игры через JavaScript-оценку страницы.
    Возвращает (state, info_str, dino_status) или (-1, причина, None) при ошибке/смерти.
    """
    try:
        result = page.evaluate(
            """
            () => {
                const runner = window.Runner && window.Runner.instance_;
                if (!runner) return { status: "NO_RUNNER" };
                if (runner.crashed) return { status: "DEAD" };
                if (!runner.playing) return { status: "NOT_PLAYING" };

                const dino = runner.tRex;
                const obstacles = (runner.horizon && runner.horizon.obstacles) || [];
                const speed = runner.currentSpeed || 6;

                // Ищем ближайшее препятствие, которое ещё впереди динозавра
                let nextObstacle = null;
                for (let i = 0; i < obstacles.length; i++) {
                    const o = obstacles[i];
                    if (o.xPos + o.width > dino.xPos - 5) {
                        nextObstacle = o;
                        break;
                    }
                }

                if (!nextObstacle) {
                    return { status: "OK", hasObstacle: false, speed: speed,
                             dinoStatus: dino.status };
                }

                const distance = nextObstacle.xPos - dino.xPos;
                const otype = (nextObstacle.typeConfig && nextObstacle.typeConfig.type) || "CACTUS";

                // Классифицируем препятствие:
                //   "jump"  - нужно прыгать (кактус любого размера или низко летящая птица)
                //   "duck"  - нужно приседать (птица летит высоко, над головой)
                let kind = "jump";
                if (otype === "PTERODACTYL") {
                    const y = nextObstacle.yPos;
                    // Птицы в trex-игре появляются на y = 50 / 75 / 100.
                    // Меньший y = выше над землёй -> приседаем под неё.
                    // Больший y = ниже к земле     -> прыгаем через неё (как через кактус).
                    if (y <= 60) {
                        kind = "duck";
                    } else {
                        kind = "jump";
                    }
                }

                return {
                    status: "OK",
                    hasObstacle: true,
                    distance: distance,
                    kind: kind,
                    obstacleType: otype,
                    obstacleY: nextObstacle.yPos,
                    speed: speed,
                    dinoStatus: dino.status
                };
            }
            """
        )

        status = result["status"]
        if status != "OK":
            return -1, status, None

        speed = max(result.get("speed", 6), 1.0)
        dino_status = result.get("dinoStatus", "RUNNING")

        if not result["hasObstacle"]:
            return 0, f"свободно, speed={speed:.1f}, dino={dino_status}", dino_status

        distance = result["distance"]
        kind = result["kind"]
        # TTC в КАДРАХ (не секундах!): speed = пиксели/кадр при ~60 FPS.
        # Прыжок длится ~43 кадра, пик ~22. Оптимально прыгать при ttc ~22-25.
        ttc_frames = distance / speed

        if ttc_frames > 50:
            dist_bucket = 1          # далеко
        elif ttc_frames > 25:
            dist_bucket = 2          # средне
        elif ttc_frames > 15:
            dist_bucket = 3          # близко РАНО - оптимально прыгать
        else:
            dist_bucket = 4          # близко ПОЗДНО - последний шанс

        kind_index = 0 if kind == "jump" else 1
        # state = (dist_bucket - 1) * 2 + kind_index + 1   ->  диапазон 1..8
        state = (dist_bucket - 1) * 2 + kind_index + 1

        info = (f"{kind}, type={result['obstacleType']}, y={result.get('obstacleY')}, "
                f"dist={distance:.0f}, ttc={ttc_frames:.1f}fr, speed={speed:.1f}, dino={dino_status}")
        return state, info, dino_status

    except Exception as error:
        return -1, f"ERROR: {error}", None


def choose_action(state, dino_status):
    """epsilon-жадный выбор действия с учётом физических ограничений динозавра."""
    global epsilon
    if TRAIN_MODE and random.random() < epsilon:
        # Случайное исследование, но прыгать можно только с земли
        choices = [ACTION_RUN, ACTION_DUCK]
        if dino_status != "JUMPING":
            choices.append(ACTION_JUMP)
        return random.choice(choices)

    qs = q_table[state]
    # В воздухе повторный прыжок невозможен -> рассматриваем только БЕЖАТЬ/ПРИСЕДАТЬ
    if dino_status == "JUMPING":
        valid = [ACTION_RUN, ACTION_DUCK]
        return max(valid, key=lambda a: qs[a])
    return int(np.argmax(qs))


def execute_action(page, action, dino_status):
    """Выполняет действие клавиатурой. Корректно удерживает/отпускает ArrowDown."""
    global duck_held
    if action == ACTION_JUMP:
        # перед прыжком обязательно отпускаем присед, иначе Space может не сработать
        if duck_held:
            page.keyboard.up("ArrowDown")
            duck_held = False
            time.sleep(0.02)
        if dino_status != "JUMPING":
            page.keyboard.press("Space")
    elif action == ACTION_DUCK:
        # удерживаем ArrowDown, чтобы динозавр оставался в приседе
        if not duck_held:
            page.keyboard.down("ArrowDown")
            duck_held = True
    else:  # ACTION_RUN
        if duck_held:
            page.keyboard.up("ArrowDown")
            duck_held = False


def compute_reward(state, action, next_state, prev_state):
    """
    Награда за переход. Масштаб намеренно маленький, чтобы Q-значения
    не росли безгранично: старый +1/шаг с GAMMA=0.9 раздувал Q до десятков
    (теоретический предел 1/(1-0.9)=10, на практике перелетает), и бот
    начинал прыгать/приседать без препятствий.
    """
    if next_state == -1:
        return -10.0  # смерть (не -100, чтобы масштаб был адекватный)

    reward = 0.1  # небольшое выживание - держит Q в районе приоров

    if action != ACTION_RUN:
        reward -= 0.05  # стоимость действия

    # Сильный штраф за прыжок/присед БЕЗ близкого препятствия -
    # чтобы бот не приучался дёргаться на пустом месте.
    if action != ACTION_RUN and state in (0, 1, 2, 3, 4):
        reward -= 1.0

    # Бонус за успешное прохождение близкого препятствия:
    # были в "близко" (5-8), а теперь препятствия нет (0)
    if prev_state in (5, 6, 7, 8) and next_state == 0:
        reward += 5.0

    return reward


def update_q_table(state, action, reward, next_state):
    current_q = q_table[state][action]
    max_next_q = 0.0 if next_state == -1 else max(q_table[next_state])
    new_q = current_q + ALPHA * (reward + GAMMA * max_next_q - current_q)
    q_table[state][action] = new_q


def start_game(page):
    """Перезапускает игру (Space) и ждёт первого валидного состояния."""
    global duck_held
    if duck_held:
        page.keyboard.up("ArrowDown")
        duck_held = False
    time.sleep(0.3)
    page.keyboard.press("Space")
    time.sleep(1.0)
    state, info, dino_status = get_state(page)
    return state, info, dino_status


def print_q_table():
    print("\nТекущая Q-таблица:")
    for state, values in q_table.items():
        best = int(np.argmax(values))
        print(f"  S{state}: [БЕЖАТЬ={values[0]:+.2f}, "
              f"ПРЫГАТЬ={values[1]:+.2f}, ПРИСЕДАТЬ={values[2]:+.2f}] -> {ACTION_NAMES[best]}")


def main():
    global epsilon
    print("Запускаем бесконечную игру (остановка: Ctrl+C в терминале)...")
    load_q_table()
    load_epsilon()

    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="chrome",
            headless=False,
            args=["--disable-gpu", "--disable-extensions"]
        )

        page = browser.new_page(viewport={"width": 800, "height": 600})
        page.goto(GAME_URL, wait_until="domcontentloaded")
        time.sleep(2.0)

        episode = 0
        best_steps = 0
        best_score = 0

        try:
            while True:
                episode += 1
                print(f"\n--- Игра №{episode} (epsilon={epsilon:.3f}) ---")

                state, info, dino_status = start_game(page)

                if state == -1:
                    print(f"Игра не запустилась ({info}). Перезагружаем...")
                    page.reload(wait_until="domcontentloaded")
                    time.sleep(2.0)
                    continue

                total_reward = 0.0
                step = 0
                prev_state = state

                # Внутренний цикл - одна жизнь
                while True:
                    step += 1
                    action = choose_action(state, dino_status)
                    if DEBUG_STEPS and (state != prev_state or action != ACTION_RUN):
                        print(f"  [s{step}] S{state}({STATE_TAGS.get(state,'?')}) -> {ACTION_NAMES[action]}")
                    execute_action(page, action, dino_status)

                    time.sleep(STEP_DELAY)
                    next_state, ninfo, next_dino_status = get_state(page)

                    reward = compute_reward(state, action, next_state, prev_state)
                    if TRAIN_MODE:
                        update_q_table(state, action, reward, next_state)

                    total_reward += reward
                    prev_state = state
                    state = next_state
                    dino_status = next_dino_status

                    if next_state == -1:
                        score = step  # грубая оценка: шаг ~= очко
                        print(f"Игра окончена. Шагов: {step}. "
                              f"Награда: {total_reward:.1f}. Причина: {ninfo}")
                        if step > best_steps:
                            best_steps = step
                            print(f"  🏆 Новый рекорд: {best_steps} шагов!")
                        break

                # Затухание exploration
                if TRAIN_MODE and epsilon > EPSILON_MIN:
                    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

                if TRAIN_MODE and episode % SAVE_EVERY == 0:
                    save_q_table()
                    save_epsilon()
                    print_q_table()

                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\nОстановка игры...")
            if TRAIN_MODE:
                save_q_table()
                save_epsilon()
            print_q_table()
            print(f"\nЛучший результат: {best_steps} шагов за {episode} игр.")
        finally:
            browser.close()


if __name__ == "__main__":
    main()
