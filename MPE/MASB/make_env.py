
def make_env(scenario_name, benchmark=True):

    from MASB.MASB.environment import MultiAgentEnv
    import MASB.MASB.scenario as scenarios

    scenario = scenarios.load(scenario_name + ".py").SimBid()
    world = scenario.make_world()
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.reward_match, scenario.observation, scenario.observation2)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env
