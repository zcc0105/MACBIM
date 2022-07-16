import numpy as np

from MADDPG_Model import MADDPGConfig
from MADDPG_Model.maddpg import MADDPG_master as MADDPG
from MADDPG_Model.memory import MultiAgentReplayBuffer
from MASB.make_env import make_env
import dataAnalysis
import time
from StageOne import utils
from StageOne.utils import obs_list_to_state_vector
from infoAlg import SeedsInfo


def MADDPG_test(date, is_evaluate, text_):
    cfg = MADDPGConfig()
    resCfg0 = SeedsInfo()
    startTime = time.time()
    env = make_env(cfg.env)
    n_agents = env.num_agents
    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(env.observation_space[i].shape[0])
    critic_dims = sum(actor_dims)

    cfg.n_actions = env.action_space[0].shape[0]
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, cfg, env)

    memory = MultiAgentReplayBuffer(cfg.replaybuffer, critic_dims, actor_dims,
                                    cfg.n_actions, n_agents, cfg.batch_size)

    total_steps = 0
    evaluate = is_evaluate
    best_reward = 0

    budget = utils.budgetGet(env)

    if evaluate:
        maddpg_agents.load_checkpoint()

    for i in range(cfg.N_GAMES):
        obs = env.reset()
        score = []
        success_reward = []
        done = [False] * n_agents
        episode_step = 0
        cgcs = 0
        while not any(done):
            actions = maddpg_agents.choose_action(obs)
            obs_, reward, done, info = env.step(actions)
            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)
            memory.store_transition(obs, state, actions, reward, obs_, state_, done)
            is_success = []
            for seed in env.landmarks:
                is_success.append(seed.is_success)
            if episode_step >= cfg.MAX_STEPS:
                done = [True] * n_agents
            if not evaluate and total_steps % cfg.l_step == 0:
                maddpg_agents.learn(memory)
            if all(is_success) and utils.rewardLimit(reward, budget):
                cgcs += 1
                success_reward.append(sum(reward))
                seeds = utils.initialCostGet(env)
                budget_left = utils.budgetLeftGet(env)
                for ij in range(len(budget_left)):
                    resCfg0.agentBudleft[ij].append(budget_left[ij])
                for ik in range(env.num_landmarks):
                    resCfg0.seedsPrice[ik].append(seeds[ik][0])
                for im in range(env.num_agents):
                    resCfg0.rewardAgent[im].append(reward[im])
                    resCfg0.successSeeds[im].append(env.agents[im].seeds)

                if sum(reward) > best_reward:
                    best_reward = sum(reward)
                    if not evaluate:
                        maddpg_agents.save_checkpoint()
            obs = obs_
            score.append(sum(reward))
            total_steps += 1
            episode_step += 1
            env.trainReset()

        if score:
            resCfg0.rewardPlatform.append(np.mean(score))
        else:
            resCfg0.rewardPlatform.append(0.0)
        if success_reward:
            resCfg0.successReward.append(np.mean(success_reward))
        else:
            resCfg0.successReward.append(0.0)
    endTime = time.time()
    print("MADDPG：")
    print(int(endTime - startTime))
    dataAnalysis.datawrite(0, date, text_, resCfg0.rewardPlatform, resCfg0.rewardAgent, resCfg0.successReward, resCfg0.seedsPrice,
                           resCfg0.agentBudleft, resCfg0.successSeeds)
