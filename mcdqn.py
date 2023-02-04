import gymnasium as gym
from tensorflow import keras
import numpy as np

model = keras.Sequential(
  [
    keras.layers.Dense(5, activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(3)
  ]
)
model.predict([[0.0, 0.0]])
model.compile(
  optimizer=keras.optimizers.Adam(),
  loss=keras.losses.huber
)

env = gym.make("MountainCar-v0", render_mode="human")
GAMMA = 0.99
EPSILON = 0.05  # ε-greedy 法のε。ときどきランダムに変なことをするための値。

x_train = None
y_train = None
obs, info = env.reset()
for i in range(1000000):
  x, v = obs
  q = model.predict(np.array([[x, v]]), verbose=0)[0]
  action = list(q).index(max(q))
  if np.random.rand() < EPSILON: action = env.action_space.sample() # ε-greedy
  print(q, action, x)
  obs, rew, term, truncated, info = env.step(action)
  #if (i % 10 == 0): env.render()
  x1, v1 = obs
  q1 = model.predict(np.array([[x1, v1]]), verbose=0)[0]
  teacher = q
  teacher[action] = rew + GAMMA * max(q1)
  if x_train is None:
    x_train = np.array([[x, v]])
    y_train = np.array([teacher])
  else:
    x_train = np.vstack([x_train, [x, v]])
    y_train = np.vstack([y_train, teacher])

  if term or truncated:    
    shuffled = np.arange(len(x_train))
    np.random.shuffle(shuffled)
    x_batch = x_train[shuffled]
    y_batch = y_train[shuffled]
    model.fit(x_batch, y_batch, epochs=50, verbose=0)
    keras.backend.clear_session()
    x_train = None
    y_train = None
    obs, info = env.reset()
env.close()
