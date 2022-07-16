#coding:UTF-8
from DDPG_Model import DDPGConfig
from DDPG_Model.ddpg import DDPG
import numpy as np
from MASB.make_env import make_env
import dataAnalysis
import time
from StageOne import utils
from infoAlg import SeedsInfo


def DDPG_test(date, is_evaluate, text_):
    cfg = DDPGConfig()
    resCfg1 = SeedsInfo()
    startTime = time.time()
    env = make_env(cfg.env)
    n_agents = env.num_agents
    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(env.observation_space[i].shape[0])

    cfg.n_actions = env.action_space[0].shape[0]
    ddpg_agents = DDPG(cfg, env, actor_dims, n_agents)
    budget = utils.budgetGet(env)

    total_steps = 0
    evaluate = is_evaluate
    best_reward = 0

    if evaluate:
        ddpg_agents.load_checkpoint()

    for i in range(cfg.N_GAMES):
        obs = env.reset()
        score = []
        success_reward = []
        done = [False] * n_agents
        episode_step = 0
        while not any(done):
            actions = ddpg_agents.choose_action(obs)
            obs_, reward, done, info = env.step(actions)
            ddpg_agents.remember(obs, actions, reward, obs_, done)
            is_success = []
            for seed in env.landmarks:
                is_success.append(seed.is_success)
            if total_steps % 5 == 0 and not evaluate:
                ddpg_agents.learn()
            if episode_step >= cfg.MAX_STEPS:
                done = [True] * n_agents
            if total_steps % cfg.l_step == 0 and not evaluate:
                ddpg_agents.learn()
            if all(is_success) and utils.rewardLimit(reward, budget):
                success_reward.append(sum(reward))
                seeds = utils.initialCostGet(env)
                budget_left = utils.budgetLeftGet(env)
                for ij in range(len(budget_left)):
                    resCfg1.agentBudleft[ij].append(budget_left[ij])
                for ik in range(env.num_landmarks):
                    resCfg1.seedsPrice[ik].append(seeds[ik][0])
                for im in range(env.num_agents):
                    resCfg1.rewardAgent[im].append(reward[im])
                    resCfg1.successSeeds[im].append(env.agents[im].seeds)

                if sum(reward) > best_reward:
                    best_reward = sum(reward)
                    if not evaluate:
                        ddpg_agents.save_checkpoint()
            obs = obs_
            score.append(sum(reward))
            total_steps += 1
            episode_step += 1
            env.trainReset()
        if score:
            resCfg1.rewardPlatform.append(np.mean(score))
        else:
            resCfg1.rewardPlatform.append(0.0)
        if success_reward:
            resCfg1.successReward.append(np.mean(success_reward))
        else:
            resCfg1.successReward.append(0.0)

    endTime = time.time()
    print("DDPG:")
    print(int(endTime - startTime))
    dataAnalysis.datawrite(1, date, text_, resCfg1.rewardPlatform, resCfg1.rewardAgent, resCfg1.successReward, resCfg1.seedsPrice,
                           resCfg1.agentBudleft, resCfg1.successSeeds)
