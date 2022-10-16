import numpy as np
from MPE.TD3_Model import TD3Config
from MPE.TD3_Model.TD3 import TD3
from MPE.MASB.make_env import make_env
import dataAnalysis
import time
from StageOne import utils


def TD3_test(seedsIno, date, is_evaluate, text_):
    cfg = TD3Config()
    resCfg5 = seedsIno
    startTime = time.time()
    print("TD3编号为" + date + "k=" + str(resCfg5.num_agents) + "|S|=" + str(resCfg5.num_seeds) + "episodes=" +
          str(resCfg5.N_GAMES * resCfg5.MAX_STEPS) + "实验开始")
    env = make_env(cfg.env, resCfg5)
    n_agents = env.num_agents
    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(env.observation_space[i].shape[0])

    cfg.n_actions = env.action_space[0].shape[0]

    TD3_agents = TD3(actor_dims, n_agents, cfg, env)

    total_steps = 0
    evaluate = is_evaluate
    best_reward = 0
    is_load = False

    budget = utils.budgetGet(env)

    if evaluate:
        TD3_agents.load_checkpoint()

    for i in range(resCfg5.N_GAMES):
        obs = env.reset()
        score = []
        success_reward = []
        done = [False] * n_agents
        episode_step = 0
        while not any(done):
            actions = TD3_agents.choose_action(obs)
            obs_, reward, done, info = env.step(actions)
            TD3_agents.remember(obs, actions, obs_, reward, done)
            is_success = []
            for seed in env.landmarks:
                is_success.append(seed.is_success)
            if episode_step >= resCfg5.MAX_STEPS:
                done = [True] * n_agents
            if not evaluate and total_steps % cfg.l_step == 0:
                TD3_agents.learn()

            cost = utils.costGet(env)
            # budget = utils.budgetGet(env)
            if utils.GEI(reward, cost):
            # if utils.unitCost(reward, cost):
            # if all(is_success) and utils.rewardLimit(reward, budget):
            # if utils.unitBudget(reward, budget): # 此时不关注于所有种子都拍卖掉，而是关注于公平性的情况下各项指标的情况
                is_load = True
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
            resCfg5.score.extend(score)
        else:
            resCfg5.rewardPlatform.append(0.0)
        if success_reward:
            resCfg5.successReward.append(np.mean(success_reward))
        else:
            resCfg5.successReward.append(0.0)

    endTime = time.time()
    print("TD3编号为" + date + "k=" + str(resCfg5.num_agents) + "|S|=" + str(resCfg5.num_seeds) + "episodes=" +
          str(resCfg5.N_GAMES * resCfg5.MAX_STEPS) + "实验结束")
    print("该实验耗时" + str(endTime - startTime))
    print("-------------------------------------")
    dataAnalysis.datawrite(5, date, text_, resCfg5.rewardPlatform, resCfg5.rewardAgent, resCfg5.successReward, resCfg5.seedsPrice,
                           resCfg5.agentBudleft, resCfg5.successSeeds, resCfg5.score)
    return is_load
