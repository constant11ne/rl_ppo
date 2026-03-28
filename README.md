# PPO for Continuous Control in MuJoCo

Реализация **Proximal Policy Optimization (PPO)** для задачи непрерывного управления в среде MuJoCo с использованием **Gymnasium** и **PyTorch**.  
В ноутбуке собран полный пайплайн обучения: векторизованные окружения, actor-critic policy, **GAE**, minibatch sampling, PPO update, evaluation и бонусная **recurrent PPO** с LSTM.

---

## Что здесь реализовано

Проект покрывает полный цикл обучения PPO-агента:

- создание и нормализация векторизованных окружений;
- actor-critic модель для continuous action space;
- policy wrapper для сэмплирования действий и вычисления log-probabilities;
- `EnvRunner` для сбора траекторий;
- **Generalized Advantage Estimation (GAE)**;
- нормализация advantages;
- `TrajectorySampler` для повторного прохода по одной и той же траектории в течение нескольких эпох;
- реализация **PPO clipped objective**;
- clipped value loss;
- обучение и визуализация кривой reward;
- evaluation обученного агента;
- бонусная recurrent-версия PPO с **LSTM**.

---

## Используемый стек

- **Python**
- **PyTorch**
- **Gymnasium**
- **MuJoCo**
- **NumPy**
- **Matplotlib**
- **tqdm**
- **Jupyter Notebook**

---

## Архитектура

### 1. Векторизованная среда

Для ускорения сбора опыта используется `AsyncVectorEnv`.  
На окружение дополнительно навешиваются стандартные PPO-friendly wrappers:

- `RecordEpisodeStatistics`
- `NormalizeObservation`
- `NormalizeReward`
- `ClipReward`

Это дает:
- стабилизацию масштаба наблюдений,
- стабилизацию масштаба reward,
- более устойчивое обучение.

---

### 2. Policy / Value model

Используется actor-critic архитектура с двумя отдельными MLP:

- policy network
- value network

Обе сети имеют вид:

- 2 скрытых слоя,
- по 64 нейрона,
- активация `tanh`.

#### Policy head
Policy выдает параметры нормального распределения действий:

$$
\pi_\theta(a \mid s) = \mathcal{N}(\mu_\theta(s), \Sigma_\theta(s))
$$

где:
- $\mu_\theta(s)$ — среднее,
- $\Sigma_\theta(s)$ — диагональная ковариационная матрица.

Стандартное отклонение получается через `softplus`, чтобы гарантировать положительность:

$$
\sigma_\theta(s) = softplus(x) + \varepsilon
$$

#### Value head
Value network аппроксимирует функцию ценности состояния:

$$
V_\phi(s)
$$

---

## Generalized Advantage Estimation (GAE)

Для уменьшения дисперсии policy gradient используется **GAE**.

Сначала считаются TD-residuals:

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

Затем advantage считается рекурсивно:

$$
A_t = \delta_t + \gamma \lambda (1 - d_t) A_{t+1}
$$

где:
- $\gamma$ — discount factor,
- $\lambda$ — параметр GAE,
- $d_t$ — индикатор завершения эпизода.

Target для value function:

$$
\hat{V}_t = A_t + V(s_t)
$$

После этого advantages нормализуются:

$$
\tilde{A}_t = \frac{A_t - \mu(A)}{\sigma(A) + \varepsilon}
$$

Это улучшает стабильность оптимизации.

---

## PPO objective

### Policy loss

PPO использует отношение новых и старых вероятностей действия:

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

Clipped surrogate objective:

$$
L^{\text{CLIP}}(\theta) =
\mathbb{E}_t \left[
\min \left(
r_t(\theta) A_t,\,
clip(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t
\right)
\right]
$$

В коде минимизируется отрицание этого функционала.

---

### Value loss

Используется clipped value loss:

$$
V_t^{\text{clip}} = V_{\text{old}}(s_t) +
clip(V_\theta(s_t) - V_{\text{old}}(s_t), -\epsilon, \epsilon)
$$

$$
L^{VF} =
\mathbb{E}_t \left[
\max \left(
(V_\theta(s_t) - \hat{V}_t)^2,\,
(V_t^{\text{clip}} - \hat{V}_t)^2
\right)
\right]
$$

---

### Общая функция потерь

Итоговая функция:

$$
L = L^{\pi} + c_v L^{VF}
$$

где:
- $L^{\pi}$ — policy loss,
- $L^{VF}$ — value loss,
- $c_v$ — коэффициент при value loss.

Дополнительно используется gradient clipping:

$$|\nabla_\theta L| \leq \text{max\_grad\_norm}$$

---

## Sampling pipeline

Обучение устроено так:

1. `EnvRunner` собирает траекторию длины `num_runner_steps`.
2. К траектории применяются преобразования:
   - `AsArray`
   - `GAE`
3. `TrajectorySampler`:
   - разворачивает траекторию по batch/time,
   - перемешивает её,
   - режет на minibatches,
   - позволяет проходить по одной и той же траектории несколько эпох.
4. PPO делает update по minibatches.

Это и есть одна из ключевых идей PPO:  
**использовать одну и ту же траекторию несколько раз**, но ограничивать величину policy update через clipping.

---

## Основные компоненты ноутбука

### `PolicyModel`
MLP actor-critic модель:
- policy head для параметров действия,
- value head для оценки состояния.

### `Policy`
Обертка над моделью, которая:
- сэмплирует действия;
- считает log-probabilities;
- отдает distribution и value prediction для training mode.

### `EnvRunner`
Собирает rollout из среды.

### `GAE`
Считает:
- `advantages`
- `value_targets`

### `NormalizeAdvantages`
Нормализует преимущества.

### `TrajectorySampler`
Создает minibatches для PPO и позволяет несколько эпох обучаться на одном rollout.

### `PPO`
Содержит:
- `policy_loss`
- `value_loss`
- `loss`
- `step`

### `evaluate`
Запускает trained agent без обучения и возвращает reward.

---

## Гиперпараметры

В ноутбуке используются стандартные PPO-настройки:

- `gamma = 0.99`
- `lambda = 0.95`
- `cliprange = 0.2`
- `lr = 3e-4`
- `eps = 1e-5` для Adam
- `max_grad_norm = 0.5`
- несколько parallel environments
- несколько эпох обучения на одной траектории
- minibatch training

Также используется линейный learning rate decay.

---

## Bonus: Recurrent PPO

В бонусной части реализована recurrent-версия PPO.

### Что добавляется
- `RecurrentPolicyModel`
- `RecurrentPolicy`
- `GAE_RNN`
- `EnvRunnerRNN`
- `RecurrentTrajectorySampler`
- `PPORecurrent`

### Архитектура
Recurrent model использует:

- MLP feature extractor,
- `LSTMCell`,
- policy head,
- value head.

Для последовательности наблюдений:

$$
h_t, c_t = LSTMCell(x_t, (h_{t-1}, c_{t-1}))
$$

$$
\pi_\theta(a_t \mid s_t, h_t), \quad V_\phi(s_t, h_t)
$$

При reset окружения hidden state маскируется, чтобы не протекала информация между разными эпизодами.

### Зачем это нужно
Обычный PPO с MLP видит только текущее состояние.  
Recurrent PPO полезен, если:
- наблюдения частично неполные,
- важен временной контекст,
- есть скрытая динамика, которую трудно восстановить по одному кадру.

---

## Как запустить

### Локально

Установить зависимости:

```bash
pip install gymnasium mujoco torch numpy matplotlib tqdm
````

Открыть ноутбук:

```bash
jupyter notebook rlhw4.ipynb
```

И выполнять ячейки по порядку.

---

## Что получается в итоге

После обучения агент должен научиться стабильно решать задачу continuous control в MuJoCo.
В ноутбуке строится график reward по мере роста числа взаимодействий со средой, а затем агент оценивается отдельно в режиме inference.

---

## Что здесь хорошего

Этот ноутбук хорош тем, что здесь PPO не спрятан за готовой библиотекой, а реализован руками.
То есть можно нормально проследить всю механику:

* как собираются rollout’ы,
* как считаются advantages,
* как работает clipped objective,
* как организован многократный проход по одной траектории,
* как PPO расширяется до recurrent-версии.

Именно поэтому это полезная учебная реализация, а не просто “запустил Stable-Baselines и нарисовал график”.

---

## Ограничения

У реализации есть естественные ограничения:

* это учебный ноутбук, а не production RL framework;
* мало инженерных оптимизаций;
* нет полноценной системы логирования экспериментов;
* нет отдельного конфиг-файла и модульной упаковки;
* evaluation и training tightly coupled к ноутбучному формату.

---

## Возможные улучшения

Что можно сделать дальше:

* вынести код из ноутбука в Python-модули;
* добавить логирование через TensorBoard / Weights & Biases;
* добавить entropy bonus и его сравнение;
* сделать сравнение PPO vs Recurrent PPO на одной и той же среде;
* добавить графики:

  * policy loss,
  * value loss,
  * entropy,
  * grad norm,
  * explained variance / (R^2);
* протестировать больше MuJoCo-сред;
* добавить checkpointing и resume training.

---

## Итог

Проект представляет собой полную реализацию **PPO для continuous action spaces** в MuJoCo с использованием **PyTorch** и **Gymnasium**, включая все основные компоненты современного on-policy actor-critic пайплайна: rollout collection, GAE, advantage normalization, clipped objective, minibatch updates и evaluation.
Дополнительно в ноутбуке реализована **recurrent PPO** с LSTM, что делает работу сильнее обычной базовой реализации и показывает понимание не только стандартного PPO, но и его расширений.

```
