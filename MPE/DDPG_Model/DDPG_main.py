#coding:UTF-8
from MPE.DDPG_Model import DDPGConfig
from MPE.DDPG_Model.ddpg import DDPG
import numpy as np
from MPE.MASB.make_env import make_env
import dataAnalysis
import time
from StageOne import utils


def DDPG_test(seedsIno, date, is_evaluate, text_):
    cfg = DDPGConfig()
    resCfg1 = seedsIno
    startTime = time.time()
    print("DDPG Number:" + date + " k=" + str(resCfg1.num_agents) + " |S|=" + str(resCfg1.num_seeds) + " episodes=" +
          str(resCfg1.N_GAMES * resCfg1.MAX_STEPS) + " start")
    env = make_env(cfg.env, resCfg1)
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
    is_load = False

    if evaluate:
        ddpg_agents.load_checkpoint()

    for i in range(resCfg1.N_GAMES):
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
            if episode_step >= resCfg1.MAX_STEPS:
                done = [True] * n_agents
            if total_steps % cfg.l_step == 0 and not evaluate:
                ddpg_agents.learn()

            cost = utils.costGet(env)
            if utils.GEI(reward, cost):
            # if utils.unitCost(reward, cost):
            # if all(is_success) and utils.rewardLimit(reward, budget):
            # if utils.unitBudget(reward, budget): # 此时不关注于所有种子都拍卖掉，而是关注于公平性的情况下各项指标的情况
                is_load = True
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
            resCfg1.score.extend(score)
        else:
            resCfg1.rewardPlatform.append(0.0)
        if success_reward:
            resCfg1.successReward.append(np.mean(success_reward))
        else:
            resCfg1.successReward.append(0.0)

    endTime = time.time()
    print("DDPG Number:" + date + " k=" + str(resCfg1.num_agents) + " |S|=" + str(resCfg1.num_seeds) + " episodes=" +
          str(resCfg1.N_GAMES * resCfg1.MAX_STEPS) + " end")
    print("Running time:" + str(endTime - startTime))
    print("-------------------------------------")
    dataAnalysis.datawrite(1, date, text_, resCfg1.rewardPlatform, resCfg1.rewardAgent, resCfg1.successReward, resCfg1.seedsPrice,
                           resCfg1.agentBudleft, resCfg1.successSeeds, resCfg1.score)
    return is_load
