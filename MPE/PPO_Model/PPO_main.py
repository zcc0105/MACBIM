from PPO_Model import PPOConfig
from PPO_Model.ppo import PPO
from MASB.make_env import make_env
import numpy as np
import dataAnalysis
import time
from StageOne import utils
from infoAlg import SeedsInfo


def PPO_test(date, is_evaluate, text_):
    startTime = time.time()
    cfg = PPOConfig()
    resCfg3 = SeedsInfo()
    env = make_env(cfg.env)
    n_agents = env.num_agents
    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(env.observation_space[i].shape[0])

    cfg.n_actions = env.action_space[0].shape[0]
    ppo_agents = PPO(actor_dims, cfg, env, n_agents)

    total_steps = 0
    evaluate = is_evaluate
    best_reward = 0

    budget = utils.budgetGet(env)
    if evaluate:
        ppo_agents.load()

    for i in range(cfg.N_GAMES):
        state = env.reset()
        score = []
        success_reward = []
        done = [False] * n_agents
        episode_step = 0
        if i % cfg.p_step == 0:
            print("第%d轮:" % i)
        while not any(done):
            actions, reactions, probs, vals = ppo_agents.choose_action(state)
            state_, reward, done, info = env.step(reactions)
            ppo_agents.remember(state_, actions, probs, vals, reward, done)
            is_success = []
            for seed in env.landmarks:
                is_success.append(seed.is_success)

            if episode_step >= cfg.MAX_STEPS:
                done = [True] * n_agents
            if total_steps % cfg.l_step == 0 and not evaluate:
                ppo_agents.learn()
            if all(is_success) and utils.rewardLimit(reward, budget):
                success_reward.append(sum(reward))
                seeds = utils.initialCostGet(env)
                budget_left = utils.budgetLeftGet(env)
                for ij in range(len(budget_left)):
                    resCfg3.agentBudleft[ij].append(budget_left[ij])
                for ik in range(env.num_landmarks):
                    resCfg3.seedsPrice[ik].append(seeds[ik][0])
                for im in range(env.num_agents):
                    resCfg3.rewardAgent[im].append(reward[im])
                    resCfg3.successSeeds[im].append(env.agents[im].seeds)

                if sum(reward) > best_reward:
                    best_reward = sum(reward)
                    if not evaluate:
                        ppo_agents.save()
            state = state_
            score.append(sum(reward))
            total_steps += 1
            episode_step += 1
            env.trainReset()

        if score:
            resCfg3.rewardPlatform.append(np.mean(score))
        else:
            resCfg3.rewardPlatform.append(0.0)
        if success_reward:
            resCfg3.successReward.append(np.mean(success_reward))
        else:
            resCfg3.successReward.append(0.0)

    endTime = time.time()
    print("PPO:")
    print(int(endTime - startTime))
    dataAnalysis.datawrite(3, date, text_, resCfg3.rewardPlatform, resCfg3.rewardAgent, resCfg3.successReward, resCfg3.seedsPrice,
                           resCfg3.agentBudleft, resCfg3.successSeeds)
