from MADDPG_Model.MADDPG_main import MADDPG_test
from DDPG_Model.DDPG_main import DDPG_test
from PPO_Model.PPO_main import PPO_test
from TD3_Model.TD3_main import TD3_test
from SAC_Model.SAC_main import SAC_test
import dataAnalysis
from infoAlg import SeedsInfo
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

if __name__ == '__main__':
    algNum = ['1', '2', '3', '4']
    date = [110, 1101]
    num = ['001', '011', '021', '031', '041', '051', '061', '071', '081', '091', '101',
           '111', '121', '131', '141', '151', '161']
    model = ['0', '1', '3', '5', '6']
    is_evaluate = [False, True]
    text_ = 't'
    seedsIno = SeedsInfo()
    numAgents = seedsIno.num_agents
    numSeeds = seedsIno.num_seeds
    for i in range(2):
        MADDPG_test(date[i], is_evaluate[i], text_)
        M0_RP, M0_RA, M0_SR = dataAnalysis.resGet(0, date[i], text_, numAgents, numSeeds)

        DDPG_test(date[i], is_evaluate[i], text_)
        M1_RP, M1_RA, M1_SR = dataAnalysis.resGet(1, date[i], text_, numAgents, numSeeds)

        PPO_test(date[i], is_evaluate[i], text_)
        M3_RP, M3_RA, M3_SR = dataAnalysis.resGet(3, date[i], text_, numAgents, numSeeds)
        #
        TD3_test(date[i], is_evaluate[i], text_)
        M5_RP, M5_RA, M5_SR = dataAnalysis.resGet(5, date[i], text_, numAgents, numSeeds)

        SAC_test(date[i], is_evaluate[i], text_)
        M6_RP, M6_RA, M6_SR = dataAnalysis.resGet(6, date[i], text_, numAgents, numSeeds)
    #     #
    # for i in range(5):
    #     MADDPG_test(date[1], is_evaluate[1], text_)
    #     M0_RP, M0_RA, M0_SR = dataAnalysis.resGet(0, date[1], text_, numAgents, numSeeds)
    # for i in range(2):
    # for i in range(4):
    # for m in model:
    #     name1 = 'M'+str(m)+'_ave_successBid_reward'+str(1011)+'_'+str(text_)+'.txt'
    #     M_RP = dataAnalysis.dataRead(name1)
    #     print(max(M_RP))

    name = algNum[0] + num[10]
    for item in model:
        dataAnalysis.resRe(item, name, 300, 10, 2)
    print('---')

    # dataAnalysis.modelRewardLineCharts(M0_RP, M0_RA, M0_SR, M1_RP, M1_RA, M1_SR,
    #                                    M3_RP, M3_RA, M3_SR, M5_RP, M5_RA, M5_SR,
    #                                    M6_RP, M6_RA, M6_SR)


