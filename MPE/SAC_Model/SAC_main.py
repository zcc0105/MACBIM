import numpy as np

from MPE.SAC_Model import SACConfig
from MPE.SAC_Model.SAC import SAC
from MPE.MASB.make_env import make_env
import dataAnalysis
import time
from StageOne import utils


def SAC_test(seedsIno, date, is_evaluate, text_):
    cfg = SACConfig()
    resCfg6 = seedsIno
    startTime = time.time()
    print("SAC编号为" + date + "k=" + str(resCfg6.num_agents) + "|S|=" + str(resCfg6.num_seeds) + "episodes=" +
          str(resCfg6.N_GAMES * resCfg6.MAX_STEPS) + "实验开始")
    env = make_env(cfg.env, resCfg6)
    n_agents = env.num_agents
    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(env.observation_space[i].shape[0])

    cfg.n_actions = env.action_space[0].shape[0]
    SAC_agents = SAC(actor_dims, n_agents, cfg, env)

    total_steps = 0
    evaluate = is_evaluate
    best_reward = 0
    is_load = False

    budget = utils.budgetGet(env)

    if evaluate:
        SAC_agents.load_checkpoint()

    for i in range(resCfg6.N_GAMES):
        obs = env.reset()
        score = []
        success_reward = []
        done = [False] * n_agents
        episode_step = 0
        while not any(done):
            actions = SAC_agents.choose_action(obs)
            obs_, reward, done, info = env.step(actions)
            SAC_agents.remember(obs, actions, reward, obs_, done)
            is_success = []
            for seed in env.landmarks:
                is_success.append(seed.is_success)
            if episode_step >= resCfg6.MAX_STEPS:  # MAX_STEPS = 50
                done = [True] * n_agents
            if not evaluate and total_steps % cfg.l_step == 0:
                SAC_agents.learn()

            cost = utils.costGet(env)
            if utils.GEI(reward, cost):
                is_load = True
                success_reward.append(sum(reward))
                seeds = utils.initialCostGet(env)
                budget_left = utils.budgetLeftGet(env)
                for ij in range(len(budget_left)):
                    resCfg6.agentBudleft[ij].append(budget_left[ij])
                for ik in range(env.num_landmarks):
                    resCfg6.seedsPrice[ik].append(seeds[ik][0])
                for im in range(env.num_agents):
                    resCfg6.rewardAgent[im].append(reward[im])
                    resCfg6.successSeeds[im].append(env.agents[im].seeds)

                if sum(reward) > best_reward:
                    best_reward = sum(reward)
                    if not evaluate:
                        SAC_agents.save_checkpoint()
            obs = obs_
            score.append(sum(reward))
            total_steps += 1
            episode_step += 1
            env.trainReset()

        if score:
            resCfg6.rewardPlatform.append(np.mean(score))
            resCfg6.score.extend(score)
        else:
            resCfg6.rewardPlatform.append(0.0)
        if success_reward:
            resCfg6.successReward.append(np.mean(success_reward))
        else:
            resCfg6.successReward.append(0.0)

    endTime = time.time()
    print("SAC编号为" + date + "k=" + str(resCfg6.num_agents) + "|S|=" + str(resCfg6.num_seeds) + "episodes=" +
          str(resCfg6.N_GAMES * resCfg6.MAX_STEPS) + "实验结束")
    print("该实验耗时" + str(endTime - startTime))
    print("-------------------------------------")
    dataAnalysis.datawrite(6, date, text_, resCfg6.rewardPlatform, resCfg6.rewardAgent, resCfg6.successReward, resCfg6.seedsPrice,
                           resCfg6.agentBudleft, resCfg6.successSeeds, resCfg6.score)
    return is_load
