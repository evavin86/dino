import os
import time
import random

import numpy as np
from playwright.sync_api import sync_playwright


GAME_URL = "https://wayou.github.io/t-rex-runner/"

ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.05
STEP_DELAY = 0.03
MODEL_FILE = "q_table_inf.npy"
TRAIN_MODE = True
LOAD_MODEL = True
SAVE_MODEL = True

q_table = {
    0: [5.0, -5.0],
    1: [0.0, 5.0],
    2: [-10.0, 15.0],
}


def load_q_table():
    global q_table
    if LOAD_MODEL and os.path.exists(MODEL_FILE):
        q_table = np.load(MODEL_FILE, allow_pickle=True).item()
        print("Загружена сохранённая Q-таблица.")
    else:
        print("Q-таблица создана с нуля.")


def save_q_table():
    if SAVE_MODEL:
        np.save(MODEL_FILE, q_table)
        print(f"Q-таблица сохранена в файл {MODEL_FILE}")
def get_state(page):
    try:
        result = page.evaluate(
            """
            () => {
                const runner = window.Runner?.instance_;
                if (!runner) return { status: "NO_RUNNER" };
                if (runner.crashed) return { status: "DEAD" };
                if (!runner.playing) return { status: "NOT_PLAYING" };

                const dino = runner.tRex;
                const obstacles = runner.horizon?.obstacles || [];

                let nextObstacle = null;
                for (let i = 0; i < obstacles.length; i++) {
                    if (obstacles[i].xPos + obstacles[i].width > dino.xPos) {
                        nextObstacle = obstacles[i];
                        break;
                    }
                }

                if (!nextObstacle) {
                    return { status: "OK", distance: 1000 };
                }

                const distance = nextObstacle.xPos - dino.xPos;
                return { status: "OK", distance: distance };
            }
            """
        )

        status = result["status"]
        if status != "OK":
            return -1, status

        distance = result["distance"]

        if distance > 150:
            state = 0
        elif distance > 70:
            state = 1
        else:
            state = 2

        return state, f"distance={distance:.1f}"

    except Exception as error:
        return -1, f"ERROR: {error}"


def choose_action(state):
    if TRAIN_MODE and random.random() < EPSILON:
        return random.choice([0, 1])
    return int(np.argmax(q_table[state]))


def update_q_table(state, action, reward, next_state):
    current_q = q_table[state][action]
    max_next_q = 0 if next_state == -1 else max(q_table[next_state])
    new_q = current_q + ALPHA * (reward + GAMMA * max_next_q - current_q)
    q_table[state][action] = new_q
def start_game(page):
    time.sleep(0.3)
    page.keyboard.press("Space")
    time.sleep(1.0)
    return get_state(page)


def print_q_table():
    print("\nТекущая Q-таблица:")
    for state, values in q_table.items():
        best_action = int(np.argmax(values))
        action_name = "НЕ ПРЫГАТЬ" if best_action == 0 else "ПРЫГАТЬ"
        print(f"Состояние {state}: [{values[0]:.2f}, {values[1]:.2f}] -> {action_name}")


def main():
    print("Запускаем бесконечную игру (остановка: Ctrl+C в терминале)...")
    load_q_table()

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

        try:
            while True:
                episode += 1
                print(f"\n--- Игра №{episode} ---")

                state, info = start_game(page)

                if state == -1:
                    print(f"Игра не запустилась ({info}). Перезагружаем...")
                    page.reload(wait_until="domcontentloaded")
                    time.sleep(2.0)
                    continue

                total_reward = 0
                step = 0

                while True:
                    step += 1
                    action = choose_action(state)

                    if action == 1:
                        page.keyboard.press("Space")

                    time.sleep(STEP_DELAY)
                    next_state, info = get_state(page)

                    if next_state == -1:
                        reward = -100
                    else:
                        reward = 1

                    if TRAIN_MODE:
                        update_q_table(state, action, reward, next_state)

                    total_reward += reward
                    state = next_state

                    if next_state == -1:
                        print(f"Игра окончена. Шагов: {step}. Награда: {total_reward}. Причина: {info}")
                        break

                if TRAIN_MODE and episode % 10 == 0:
                    save_q_table()
                    print_q_table()

                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\nОстановка игры...")
            if TRAIN_MODE:
                save_q_table()
            print_q_table()
        finally:
            browser.close()


if __name__ == "__main__":
    main()