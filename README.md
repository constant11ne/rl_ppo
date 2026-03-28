# Reinforcement Learning HW4

Проект посвящён реализации и анализу базовых алгоритмов обучения с подкреплением.  
Основная цель — понять, как агент обучается принимать решения, максимизируя ожидаемое вознаграждение в среде.

---

## Постановка задачи

Рассматривается стандартная MDP (Markov Decision Process):

- состояния: \( s \in \mathcal{S} \)
- действия: \( a \in \mathcal{A} \)
- награда: \( r(s, a) \)
- политика: \( \pi(a|s) \)

Цель агента — максимизировать ожидаемое суммарное вознаграждение:

\[
J(\pi) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]
\]

где \( \gamma \in [0,1] \) — коэффициент дисконтирования.

---

## Функции ценности

### Value function

\[
V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s \right]
\]

### Action-value function

\[
Q^\pi(s,a) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s, a_0 = a \right]
\]

---

## Bellman уравнения

### Для \( V^\pi \)

\[
V^\pi(s) = \mathbb{E}_{a \sim \pi} \left[ r(s,a) + \gamma \mathbb{E}_{s'} V^\pi(s') \right]
\]

### Для \( Q^\pi \)

\[
Q^\pi(s,a) = r(s,a) + \gamma \mathbb{E}_{s'} \mathbb{E}_{a' \sim \pi} Q^\pi(s',a')
\]

---

## Алгоритмы

### 1. Value Iteration

Итеративное обновление:

\[
V_{k+1}(s) = \max_a \left[ r(s,a) + \gamma \mathbb{E}_{s'} V_k(s') \right]
\]

---

### 2. Policy Iteration

Состоит из двух шагов:

**Policy Evaluation:**

\[
V^\pi(s) \leftarrow \mathbb{E}_{a \sim \pi} \left[ r(s,a) + \gamma \mathbb{E}_{s'} V^\pi(s') \right]
\]

**Policy Improvement:**

\[
\pi'(s) = \arg\max_a Q^\pi(s,a)
\]

---

### 3. Q-learning

Off-policy алгоритм:

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \left( r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right)
\]

---

### 4. SARSA

On-policy алгоритм:

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \left( r + \gamma Q(s',a') - Q(s,a) \right)
\]

---

## Исследования

В ноутбуке проведены эксперименты:

- сравнение алгоритмов (Value Iteration / Policy Iteration / Q-learning / SARSA)
- анализ сходимости
- влияние гиперпараметров:
  - \( \gamma \)
  - \( \alpha \)
  - стратегия исследования (epsilon-greedy)

---

## Exploration vs Exploitation

Используется epsilon-greedy стратегия:

\[
a =
\begin{cases}
\text{random action}, & \text{с вероятностью } \epsilon \\
\arg\max_a Q(s,a), & \text{иначе}
\end{cases}
\]

---

## Вывод

В ходе работы реализованы и протестированы классические алгоритмы RL.  
Эксперименты показывают различия между on-policy и off-policy подходами, а также влияние гиперпараметров на скорость и стабильность обучения.

---

## Итог

Проект демонстрирует базовые принципы reinforcement learning и даёт практическое понимание того, как агенты обучаются в MDP.
