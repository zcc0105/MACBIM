from gym.envs.registration import register

register(
    id='MASB-v0',
    entry_point='MASB.MASB.envs.simple_bidding:SimBid',
    max_episode_steps=100,
)
