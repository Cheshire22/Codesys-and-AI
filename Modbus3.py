from pyModbusTCP.server import ModbusServer
import gym
import numpy as np
import time
import struct

# Инициализация сервера
server = ModbusServer(host="192.168.0.101", port=5020, no_block=True)
server.start()
print("Modbus Server started on port 5020")

# Инициализация Gym
env = gym.make('CartPole-v1', render_mode='human')
observation, _ = env.reset()

# Статистика
episode_count = 0
total_reward = 0.0
episode_rewards = []


def float_to_registers(f_val):
    """Convert float to 2 registers (little-endian)"""
    [dword] = struct.unpack('<I', struct.pack('<f', f_val))
    return [dword & 0xFFFF, (dword >> 16) & 0xFFFF]


try:
    while True:
        # 1. Подготовка данных для регистров 0-9
        reg_data = []

        # Кодируем observation (4 float32 -> 8 регистров)
        for val in observation:
            reg_data.extend(float_to_registers(float(val)))

        # Текущее вознаграждение (примерное значение)
        current_reward = 1.0
        reg_data.extend(float_to_registers(current_reward))

        # 2. Запись в регистры 0-9
        server.data_bank.set_holding_registers(0, reg_data)

        # 3. Чтение действия из регистра 10
        action_data = server.data_bank.get_holding_registers(10, 1)
        action = action_data[0] if action_data else 0

        # 4. Шаг в среде
        observation, reward, done, _, _ = env.step(action)
        total_reward += reward

        # 5. Обработка завершения эпизода
        if done:
            episode_rewards.append(total_reward)
            total_reward = 0.0
            observation, _ = env.reset()
            episode_count += 1

            # 6. Обновление статистики каждые 10 эпизодов
            if episode_count % 10 == 0:
                avg_reward = sum(episode_rewards[-10:]) / 10
                server.data_bank.set_holding_registers(20, float_to_registers(avg_reward))
                print(f"Avg reward (last 10): {avg_reward:.2f}")

        time.sleep(0.05)

except KeyboardInterrupt:
    print("Server shutting down...")
finally:
    server.stop()