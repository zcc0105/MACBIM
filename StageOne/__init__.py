#coding:UTF-8
from StageOne.reward import reward_test
from StageOne import Seeds_select
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


if __name__ == '__main__':
    date = 181
    is_evaluate = False
    text_ = 't'
    Seeds_select.seedsSelect()
