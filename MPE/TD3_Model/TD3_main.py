import numpy as np
from TD3_Model import TD3Config
from TD3_Model.TD3 import TD3
from MASB.make_env import make_env
import dataAnalysis
import time
from StageOne import utils
from infoAlg import SeedsInfo


def TD3_test(date, is_evaluate, text_):
    cfg = TD3Config()
    resCfg5 = SeedsInfo()
    startTime = time.time()
    env = make_env(cfg.env)
    n_agents = env.num_agents
    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(env.observation_space[i].shape[0])

    cfg.n_actions = env.action_space[0].shape[0]

    TD3_agents = TD3(actor_dims, n_agents, cfg, env)

    total_steps = 0
    evaluate = is_evaluate
    best_reward = 0

    budget = utils.budgetGet(env)

    if evaluate:
        TD3_agents.load_checkpoint()

    for i in range(cfg.N_GAMES):
        obs = env.reset()
        score = []
        success_reward = []
        done = [False] * n_agents
        episode_step = 0
        if i % cfg.p_step == 0:
            print("第%d轮:" % i)
        while not any(done):
            actions = TD3_agents.choose_action(obs)
            obs_, reward, done, info = env.step(actions)
            TD3_agents.remember(obs, actions, obs_, reward, done)
            is_success = []
            for seed in env.landmarks:
                is_success.append(seed.is_success)
            if episode_step >= cfg.MAX_STEPS:
                done = [True] * n_agents
            if not evaluate and total_steps % cfg.l_step == 0:
                TD3_agents.learn()
            if all(is_success) and utils.rewardLimit(reward, budget):
                success_reward.append(sum(reward))
                seeds = utils.initialCostGet(env)
                budget_left = utils.budgetLeftGet(env)
                for ij in range(len(budget_left)):
                    resCfg5.agentBudleft[ij].append(budget_left[ij])
                for ik in range(env.num_landmarks):
                    resCfg5.seedsPrice[ik].append(seeds[ik][0])
                for im in range(env.num_agents):
                    resCfg5.rewardAgent[im].append(reward[im])
                    resCfg5.successSeeds[im].append(env.agents[im].seeds)

                if sum(reward) > best_reward:
                    best_reward = sum(reward)
                    if not evaluate:
                        TD3_agents.save_checkpoint()
            obs = obs_
            score.append(sum(reward))
            total_steps += 1
            episode_step += 1
            env.trainReset()

        if score:
            resCfg5.rewardPlatform.append(np.mean(score))
        else:
            resCfg5.rewardPlatform.append(0.0)
        if success_reward:
            resCfg5.successReward.append(np.mean(success_reward))
        else:
            resCfg5.successReward.append(0.0)

    endTime = time.time()
    print("TD3:")
    print(int(endTime - startTime))
    dataAnalysis.datawrite(5, date, text_, resCfg5.rewardPlatform, resCfg5.rewardAgent, resCfg5.successReward, resCfg5.seedsPrice,
                           resCfg5.agentBudleft, resCfg5.successSeeds)
