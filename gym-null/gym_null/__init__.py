from gym.envs.registration import register

register(
    id='janest-v0',
    entry_point='gym_null.envs:JanestEnv',
)