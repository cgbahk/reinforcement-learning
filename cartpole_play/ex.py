import gym

env = gym.make('CartPole-v0')

st = env.reset()

for x in range(130):
    s = env.step(1 if st[2] > 0 else 0)
    st = s[0]
    env.render()
