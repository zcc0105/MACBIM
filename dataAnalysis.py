#coding:UTF-8
import math
from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import norm
from infoAlg import SeedsInfo


from StageOne import utils


def dataSIGMOD(data):
    data_ = []
    is_cal = False
    for item in data:
        item = 1.0/(1+np.exp(-item))
        data_.append(item)
    if is_cal:
        x_mean, x_std = dataMLE(data_)
        return seedTrans(x_mean)
    else:
        return data_


def seedTrans(x):
    x_new = -math.log((1-x)/x)
    return x_new


def dataMLE(data):
    mu = np.mean(data)
    std = np.std(data)
    x = mu + pow(std, 2)*np.random.randn(10000)
    x1, x2 = norm.fit(x)
    return round(x1, 2)


def datawrite(m, date, text_, rewardPlatform, rewardAgents, successReward, successPrices, budgetLefts, successSeeds):
    filename1 = 'M' + str(m) + '_platform_reward' + str(date) + '_' + str(text_) + '.txt'
    filename2 = 'M' + str(m) + '_ave_successBid_reward' + str(date) + '_' + str(text_) + '.txt'
    dataStore(filename1, rewardPlatform, 'reward of platform')
    dataStore(filename2, successReward, 'reward of successful bidding reward')
    for i in range(len(rewardAgents)):
        filename = 'M' + str(m) + '_agent' + str(i+1) + '_reward' + str(date) + '_' + str(text_) + '.txt'
        dataStore(filename, rewardAgents[i], 'reward of agent')
        filename = 'M' + str(m) + '_agent' + str(i+1) + '_seeds' + str(date) + '_' + str(text_) + '.txt'
        dataStore(filename, successSeeds[i], 'seeds of agent')
    for i in range(len(budgetLefts)):
        filename = 'M' + str(m) + '_agent' + str(i+1) + '_budLeft' + str(date) + '_' + str(text_) + '.txt'
        dataStore(filename, budgetLefts[i], 'left budget of agent')
    for i in range(len(successPrices)):
        filename = 'M' + str(m) + '_seed' + str(i) + '_price' + str(date) + '_' + str(text_) + '.txt'
        dataStore(filename, successPrices[i], 'successful bidding price of seed')


def dataStore(filename, data, key):
    t = ''
    storeFname = './ModelsRes/' + filename
    with open(storeFname, 'w') as f:
        f.write('# start to write %s\n' % key)
        for item in data:
            t = t + str(item)
            f.write(t)
            f.write('\n')
            t = ''
        f.close()


def dataRead(filename):
    data = []
    readFname = './ModelsRes/' + filename
    with open(readFname, 'r') as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            elif line.startswith('n'):
                data.append(0.0)
            elif line.startswith('['):
                data.append(float((line.strip())[1: len(line)-2]))
            else:
                data.append(float(line.strip()))
        f.close()
    return data


def dataReadASR(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            if len(line.strip()) == 2:
                continue
            else:
                line_ = (line.strip())[1: len(line)-2]
                data_ = line_.split(', ')
                for item in data_:
                    data.append(float(item))
    return data


def subLineCharts(data1, data2):
    plt.plot(data1, linewidth=2, linestyle='-', color='blue')
    plt.plot(data2, linewidth=2, linestyle='-.', color='red')
    plt.legend(['1', '2'])
    plt.title('Seeds Prices' , fontsize=16)
    plt.xlabel('Time', fontsize=10)
    plt.ylabel('Price', fontsize=10)
    plt.tick_params(axis='both', labelsize=10)
    plt.show()


def seedsLineCharts(price0, price1, price2, price3, price4):
    fig = plt.figure(constrained_layout=True, figsize=(8, 6))
    gs = fig.add_gridspec(2, 6)  # 创立2 * 6 网格
    axs0 = fig.add_subplot(gs[0, :2])
    axs0.plot(price0, linewidth=2)
    axs0.set_title('Seeds0 Prices', fontsize=10)
    axs0.set(xlabel='Time', ylabel='Price')
    axs1 = fig.add_subplot(gs[0, 2:4])
    axs1.plot(price1, linewidth=2)
    axs1.set_title('Seeds1 Prices', fontsize=10)
    axs1.set(xlabel='Time', ylabel='Price')
    axs2 = fig.add_subplot(gs[0, 4:6])
    axs2.plot(price2, linewidth=2)
    axs2.set_title('Seeds2 Prices', fontsize=10)
    axs2.set(xlabel='Time', ylabel='Price')
    axs3 = fig.add_subplot(gs[1, 1:3])
    axs3.plot(price3, linewidth=2)
    axs3.set_title('Seeds3 Prices', fontsize=10)
    axs3.set(xlabel='Time', ylabel='Price')
    axs4 = fig.add_subplot(gs[1, 3:5])
    axs4.plot(price4, linewidth=2)
    axs4.set_title('Seeds4 Prices', fontsize=10)
    axs4.set(xlabel='Time', ylabel='Price')
    plt.show()


def seedsScatterDiagram(price, name):
    num_seeds = len(price)
    fig = plt.figure(constrained_layout=True, figsize=(8, 6))
    gs = fig.add_gridspec(2, num_seeds+1)  # 创立2 * 6 网格
    x0 = list(range(0, len(price[0]), 1))
    for i in range(num_seeds):
        if i <= num_seeds/2:
            axs = fig.add_subplot(gs[0, i*2:2*i+2])
            axs.scatter(x0, price[i], s=10)
            axs.set_title('Seeds%d Price' % i, fontsize=10)
            axs.set(xlabel='Time', ylabel='Price')
        else:
            axs = fig.add_subplot(gs[1, i*2+1:2*i+3])
            axs.scatter(x0, price[i], s=10)
            axs.set_title('Seeds%d Price' % i, fontsize=10)
            axs.set(xlabel='Time', ylabel='Price')
    plt.suptitle(name)
    plt.show()


def annotate_axes(ax, reward, k):
    if k == 0:
        ax.set_title('Reward of Platform')
    elif k == 1:
        ax.set_title('Average reward of successful bid')
    else:
        ax.set_title('Reward of agent%d' % (k-1))
    ax.plot(reward, linewidth=1)
    ax.set(xlabel='Time', ylabel='Reward')


def agentRewardLineCharts(reward0, success_reward, reward, name):
    num_agents = len(reward)
    list_row0 = ['0' for _ in range(num_agents)]
    list_row1 = ['1' for _ in range(num_agents)]
    list_row2 = [str(i+2) for i in range(num_agents)]
    fig, axd = plt.subplot_mosaic([list_row0, list_row1, list_row2],
                                  figsize=(8, 6), constrained_layout=True)
    annotate_axes(axd['0'], reward0, 0)
    annotate_axes(axd['1'], success_reward, 1)
    for i in range(num_agents):
        annotate_axes(axd[str(i+2)], reward[i+1], i+2)
    plt.suptitle(name)
    plt.show()


def paperLine(ydata):
    fig = plt.figure(figsize=(20, 5))
    xdata = np.arange(4)
    bar_width = 0.15
    opacity = 0.8
    error_config = {'ecolor': '0.8'}
    ax1 = fig.add_subplot(141)
    ax1.bar(xdata + bar_width*0, ydata[1], bar_width,
                    alpha=opacity, color='#6A5ACD',
                    error_kw=error_config)
    ax1.bar(xdata + bar_width*1, ydata[3], bar_width,
                     alpha=opacity, color='#228b22',
                     error_kw=error_config)
    ax1.bar(xdata + bar_width*2, ydata[4], bar_width,
                     alpha=opacity, color='#4169e1',
                     error_kw=error_config)
    ax1.bar(xdata + bar_width*3, ydata[2], bar_width,
            alpha=opacity, color='#ff8c00',
            error_kw=error_config)
    ax1.bar(xdata + bar_width*4, ydata[0], bar_width,
            alpha=opacity, color='#B22222',
            error_kw=error_config)
    ax1.set_xticks(xdata + 5 * bar_width / 5)
    ax1.set_xticklabels(('1', '2', '3', '4'))
    ax1.set_xlabel('Traininf Times')
    ax1.set_ylabel('SR(%)')

    ax2 = fig.add_subplot(142)
    ax2.bar(xdata + bar_width*0, ydata[6], bar_width,
                    alpha=opacity, color='#6A5ACD',
                    error_kw=error_config)

    ax2.bar(xdata + bar_width*1, ydata[8], bar_width,
                     alpha=opacity, color='#228b22',
                     error_kw=error_config)
    ax2.bar(xdata + bar_width*2, ydata[9], bar_width,
                     alpha=opacity, color='#4169e1',
                     error_kw=error_config)
    ax2.bar(xdata + bar_width*3, ydata[7], bar_width,
            alpha=opacity, color='#ff8c00',
            error_kw=error_config)
    ax2.bar(xdata + bar_width*4, ydata[5], bar_width,
            alpha=opacity, color='#B22222',
            error_kw=error_config)
    ax2.set_xticks(xdata + 5 * bar_width / 5)
    ax2.set_xticklabels(('1', '2', '3', '4'))
    ax2.set_xlabel('Traininf Times')
    ax2.set_ylabel('SR(%)')

    ax3 = fig.add_subplot(143)
    ax3.bar(xdata + bar_width*0, ydata[11], bar_width,
                    alpha=opacity, color='#6A5ACD',
                    error_kw=error_config)
    ax3.bar(xdata + bar_width*1, ydata[13], bar_width,
                     alpha=opacity, color='#228b22',
                     error_kw=error_config)
    ax3.bar(xdata + bar_width*2, ydata[14], bar_width,
                     alpha=opacity, color='#4169e1',
                     error_kw=error_config)
    ax3.bar(xdata + bar_width*3, ydata[12], bar_width,
            alpha=opacity, color='#ff8c00',
            error_kw=error_config)
    ax3.bar(xdata + bar_width*4, ydata[10], bar_width,
            alpha=opacity, color='#B22222',
            error_kw=error_config)
    ax3.set_xticks(xdata + 5 * bar_width / 5)
    ax3.set_xticklabels(('1', '2', '3', '4'))
    ax3.set_xlabel('Traininf Times')
    ax3.set_ylabel('SR(%)')
    
    ax4 = fig.add_subplot(144)
    ax4.bar(xdata + bar_width*0, ydata[16], bar_width,
                    alpha=opacity, color='#6A5ACD',
                    error_kw=error_config)
    ax4.bar(xdata + bar_width*1, ydata[18], bar_width,
                     alpha=opacity, color='#228b22',
                     error_kw=error_config)
    ax4.bar(xdata + bar_width*2, ydata[19], bar_width,
                     alpha=opacity, color='#4169e1',
                     error_kw=error_config)
    ax4.bar(xdata + bar_width*3, ydata[17], bar_width,
            alpha=opacity, color='#ff8c00',
            error_kw=error_config)
    ax4.bar(xdata + bar_width*4, ydata[15], bar_width,
            alpha=opacity, color='#B22222',
            error_kw=error_config)
    ax4.set_xticks(xdata + 5 * bar_width / 5)
    ax4.set_xticklabels(('1', '2', '3', '4'))
    ax4.set_xlabel('Training Times')
    ax4.set_ylabel('SR(%)')

    fig1 = plt.figure(figsize=(5, 4))
    fig1.legend(['DDPG', 'SAC', 'TD3', 'PPO', 'MADDPG'], ncol=5, loc=9, frameon=False)

    plt.tight_layout()
    plt.show()
    # plt.savefig('4.jpg', dpi=450)


def modelRewardLineCharts(M0_RP, M0_RA, M0_SR, M1_RP, M1_RA, M1_SR,
                        M3_RP, M3_RA, M3_SR, M5_RP, M5_RA, M5_SR,
                        M6_RP, M6_RA, M6_SR):
    num_agents = len(M0_RA)
    plt.figure(2+num_agents, figsize=(10, 3*(2+num_agents)))
    plt.subplot(511)
    plt.plot(M0_RP, linewidth=2, color='red', linestyle='-')
    plt.plot(M1_RP, linewidth=2, color='blue', linestyle='-.')
    plt.plot(M3_RP, linewidth=2, color='green', linestyle=':')
    plt.plot(M5_RP, linewidth=2, color='yellow', linestyle='dotted')
    plt.plot(M6_RP, linewidth=2, color='c', linestyle='-')
    plt.legend(['MADDPG', 'DDPG', 'PPO', 'TD3', 'SAC'], loc='best')
    plt.title('Reward of Platform')
    plt.xlabel('Time')
    plt.ylabel('Reward')

    plt.subplot(512)
    plt.plot(M0_SR, linewidth=2, color='red', linestyle='-')
    plt.plot(M1_SR, linewidth=2, color='blue', linestyle='-.')
    plt.plot(M3_SR, linewidth=2, color='green', linestyle=':')
    plt.plot(M5_SR, linewidth=2, color='yellow', linestyle='dotted')
    plt.plot(M6_SR, linewidth=2, color='c', linestyle='-')
    plt.legend(['MADDPG', 'DDPG', 'PPO', 'TD3', 'SAC'], loc='best')
    plt.title('Average reward of successful bid')
    plt.xlabel('Time')
    plt.ylabel('Reward')

    plt.subplot(513)
    plt.plot(M0_RA[0], linewidth=2, color='red', linestyle='-')
    plt.plot(M1_RA[0], linewidth=2, color='blue', linestyle='-.')
    plt.plot(M3_RA[0], linewidth=2, color='green', linestyle=':')
    plt.plot(M5_RA[0], linewidth=2, color='yellow', linestyle='dotted')
    plt.plot(M6_RA[0], linewidth=2, color='c', linestyle='-')
    plt.legend(['MADDPG', 'DDPG', 'PPO', 'TD3', 'SAC'], loc='best')
    plt.title('Reward of Agent1')
    plt.xlabel('Time')
    plt.ylabel('Reward')

    plt.subplot(514)
    plt.plot(M0_RA[1], linewidth=2, color='red', linestyle='-')
    plt.plot(M1_RA[1], linewidth=2, color='blue', linestyle='-.')
    plt.plot(M3_RA[1], linewidth=2, color='green', linestyle=':')
    plt.plot(M5_RA[1], linewidth=2, color='yellow', linestyle='dotted')
    plt.plot(M6_RA[1], linewidth=2, color='c', linestyle='-')
    plt.legend(['MADDPG', 'DDPG', 'PPO', 'TD3', 'SAC'], loc='best')
    plt.title('Reward of Agent2')
    plt.xlabel('Time')
    plt.ylabel('Reward')

    plt.subplot(514)
    plt.plot(M0_RA[2], linewidth=2, color='red', linestyle='-')
    plt.plot(M1_RA[2], linewidth=2, color='blue', linestyle='-.')
    plt.plot(M3_RA[2], linewidth=2, color='green', linestyle=':')
    plt.plot(M5_RA[2], linewidth=2, color='yellow', linestyle='dotted')
    plt.plot(M6_RA[2], linewidth=2, color='c', linestyle='-')
    plt.legend(['MADDPG', 'DDPG', 'PPO', 'TD3', 'SAC'], loc='best')
    plt.title('Reward of Agent2')
    plt.xlabel('Time')
    plt.ylabel('Reward')

    plt.tight_layout()
    plt.show()


def avgModelReward(data: list):
    data_ = []
    for item in data:
        if item != 0:
            data_.append(item)
    if len(data_) == 0:
        return 0
    return round(sum(data_)/len(data_), 3)


def initCost(data0, data1, data2, data3, data4):
    plt.figure(1, figsize=(8, 6))
    plt.plot(data0, linewidth=2, color='red', linestyle='-.')
    plt.plot(data1, linewidth=2, color='blue', linestyle='-')
    plt.plot(data2, linewidth=2, color='green', linestyle=':')
    plt.plot(data3, linewidth=2, color='yellow', linestyle='-.')
    plt.plot(data4, linewidth=2, color='c', linestyle='solid')
    plt.legend(['8', '170', '86', '47', '239'])
    plt.title('Initial price of seeds')
    plt.ylabel('Price')
    plt.show()


def seedsKsTest(data, name):
    is_plot = False
    num_seeds = len(data)
    for i in range(num_seeds):
        sname = 'seed%s' % i
        u = np.mean(data[i])
        std = np.std(data[i])
        stateistic, pvalue = stats.kstest(data[i], 'norm', (u, std))
        if pvalue > 0.05:
            continue
        else:
            data_ = dataSIGMOD(data[i])
            u_ = np.mean(data_)
            std_ = np.std(data_)
            stateistic_, pvalue_ = stats.kstest(data_, 'norm', (u_, std_))
    if is_plot:
        fig = plt.figure(2*num_seeds, figsize=(4*num_seeds, 6))
        x0 = list(range(0, len(data[0]), 1))
        for i in range(num_seeds):
            num = 200 + num_seeds*10 + (i+1)
            plt.subplot(num)
            plt.scatter(x0, data[i])
            plt.title('Price of seed%s' % i)
            plt.ylabel('price')
        for i in range(num_seeds):
            ax = fig.add_subplot(2, num_seeds, num_seeds+i+1)
            data[i].hist(bins=100, alpha=0.99, ax=ax)
            data[i].plot(kind='kde', secondary_y=True, ax=ax)
        plt.suptitle(name)
        plt.tight_layout()
        plt.show()


def resRe(m, name, TN, budSum, numAgents):
    M_ABL = {}
    for i in range(numAgents):
        filename = 'M'+str(m)+'_agent' +str(i+1) +'_budLeft'+str(name)+'_t.txt'
        M_ABL[i] = dataRead(filename)
    times = 0
    for i in range(len(M_ABL[0])):
        sum_ = 0
        for j in range(numAgents):
            sum_ += M_ABL[j][i]
        if sum_ / budSum <= 0.35:
            times += 1
    print(times/TN)


# 读取实验结果txt文件，并返回
def resGet(m, date, text_, numAgents, numSeeds):
    cfg = SeedsInfo()
    is_print = True
    is_plot = False
    is_ksTest = True
    model = {0: 'MADDPG', 1: 'DDPG', 2 :'A2C',
             3: 'PPO', 4: 'DDQN', 5:'TD3', 6:'SAC'}
    name1 = 'M'+str(m)+'_platform_reward'+str(date)+'_'+str(text_)+'.txt'
    M_RA = {}       # 存储竞争者的收益
    M_ABL = {}      # 存储竞争者的剩余收益
    avg_RA = {}
    for i in range(numAgents):
        filename1 = 'M'+str(m)+'_agent' + str(i+1) + '_reward'+str(date)+'_'+str(text_)+'.txt'
        M_RA[i+1] = dataRead(filename1)
        filename2 = 'M'+str(m)+'_agent' +str(i+1) +'_budLeft'+str(date)+'_'+str(text_)+'.txt'
        M_ABL[i+1] = dataRead(filename2)
        avg_RA[i+1] = avgModelReward(M_RA[i+1])
    name4 = 'M'+str(m)+'_ave_successBid_reward'+str(date)+'_'+str(text_)+'.txt'
    M_SB = {}
    for i in range(numSeeds):
        filename = 'M'+str(m)+'_seed' +str(0) +'_price'+str(date)+'_'+str(text_)+'.txt'
        M_SB[i] = dataRead(filename)

    M_RP = dataRead(name1)
    M_SR = dataRead(name4)
    avg_RP = avgModelReward(M_RP)
    avg_reward = [avg_RA[i+1] for i in range(numAgents)]
    avg_SR = avgModelReward(M_SR)
    cost_RA = {}
    for i in range(numAgents):
        cost_RA[i+1] = (cfg.budget[i]-avgModelReward(M_ABL[i+1]))/cfg.budget[i]
    M_resRate = utils.fairCal(avg_reward, cfg.budget)
    if is_plot:
        seedsScatterDiagram(M_SB, model[m])
        agentRewardLineCharts(M_RP, M_SR, M_RA, model[m])
    if is_print:
        print(str(len(M_SB[0])/(cfg.MAX_STEPS*cfg.N_GAMES/100)))
        print(str(avg_RP))
        for i in range(numAgents):
            print(str(avg_RA[i+1]))
        print(str(avg_SR))
        for i in range(numAgents):
            print(str(round(cost_RA[i+1]*100, 2)))
        for i in range(len(M_resRate)):     # 输出公平率
            print(str(round(M_resRate[i]*100, 1)))
        print('---------------------------')
    if avg_RA[1] == 0 or len(M_SB[0]) == 1:
        is_ksTest = False
    if is_ksTest:
        seedsKsTest(M_SB, model[m])
    return M_RP, M_RA, M_SR


# 种子起拍价处理
def seedPriceProcess(M_SB0, M_SB1, M_SB2, M_SB3, M_SB4):
    M_SB0_arr = np.array(M_SB0)
    M_SB1_arr = np.array(M_SB1)
    M_SB2_arr = np.array(M_SB2)
    M_SB3_arr = np.array(M_SB3)
    M_SB4_arr = np.array(M_SB4)
    # 中位数
    median_SB0 = np.median(M_SB0_arr)
    median_SB1 = np.median(M_SB1_arr)
    median_SB2 = np.median(M_SB2_arr)
    median_SB3 = np.median(M_SB3_arr)
    median_SB4 = np.median(M_SB4_arr)
    
    # 均值
    mean_SB0 = np.mean(M_SB0_arr)
    mean_SB1 = np.mean(M_SB1_arr)
    mean_SB2 = np.mean(M_SB2_arr)
    mean_SB3 = np.mean(M_SB3_arr)
    mean_SB4 = np.mean(M_SB4_arr)

    # 众数
    collections_SB0 = Counter(M_SB0_arr)
    argmax_SB0 = collections_SB0.most_common(1)
    collections_SB1 = Counter(M_SB1_arr)
    argmax_SB1 = collections_SB1.most_common(1)
    collections_SB2 = Counter(M_SB2_arr)
    argmax_SB2 = collections_SB2.most_common(1)
    collections_SB3 = Counter(M_SB3_arr)
    argmax_SB3 = collections_SB3.most_common(1)
    collections_SB4 = Counter(M_SB4_arr)
    argmax_SB4 = collections_SB4.most_common(1)




    


