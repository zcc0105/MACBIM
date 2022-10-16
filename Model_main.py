import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'
import sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from MPE.MADDPG_Model.MADDPG_main import MADDPG_test
from MPE.DDPG_Model.DDPG_main import DDPG_test
from MPE.PPO_Model.PPO_main import PPO_test
from MPE.TD3_Model.TD3_main import TD3_test
from MPE.SAC_Model.SAC_main import SAC_test
import dataAnalysis
from infoAlg import SeedsInfo

if __name__ == '__main__':
    datasetNum = ['1', '2', '3', '4']
    trainNum = [['00', '01', '02', '03'],
                ['04', '05', '06', '07'],
                ['08', '09', '10', '11']
                ]
    muchAgentTrainNum = ['14', '13', '14', '15', '16']
    muchAgentNum = [5, 10, 20, 50, 100]
    seedsNum = [20, 50, 100, 200, 400]

    stepTimesNum = [1, 2, 3, 4]

    is_evaluate = [False, True]
    text_ = 't'
    seedsIno = SeedsInfo()
    dataset = [3, 0, 2, 1]

    dataset_id = 0
    im = 0
    seedsIno.re_num(dataset[dataset_id], muchAgentNum[im], seedsNum[im], stepTimesNum[3])
    dataRecordNum = [datasetNum[dataset_id] + muchAgentTrainNum[im],
                     datasetNum[dataset_id] + muchAgentTrainNum[im] + 'huge']
    for i in range(2):
        is_loadcheck0 = DDPG_test(seedsIno, dataRecordNum[i], is_evaluate[i], text_)
        if not is_loadcheck0:
            break
