import pickle
import numpy as np


env_name = 'relocate-expert'

# path = 'data/door-expert-v0.pkl'
# f = open(path, 'rb')
# data = pickle.load(f)
# id = np.random.randint(0, 1000, 100)
# trans = []
# for i in id:
#     if sum(data[i]['rewards']) >= 500:
#         trans.append(data[i])
# with open('data/door-filter-100.pkl', 'wb') as f:
#     pickle.dump(trans, f)

# ADD Q VALUE
# with open('data/door-filter-100.pkl', 'rb') as f:
with open('data/{}-100.pkl'.format(env_name), 'rb') as f:
    data = pickle.load(f)
    for i in range(len(data)):
        traj = data[i]
        data[i]['Q'] = [0] * len(traj['rewards'])
        for step in range(len(traj['rewards']) - 1, -1, -1):
            if step == len(traj['rewards']) - 1:
                data[i]['Q'][step] = data[i]['rewards'][step]
            else:
                data[i]['Q'][step] = data[i]['rewards'][step] + data[i]['Q'][step+1]*0.99
with open('data/{}-filterQ.pkl'.format(env_name), 'wb') as f:
    pickle.dump(data, f)
    print(len(data))


# 打印数据集的最大 最小 平均rewards
# with open('data/door-filter-100.pkl', 'rb') as f:
#     data = pickle.load(f)
#     print(len(data))
#     returns = np.array([np.sum(p['rewards']) for p in data])
#     num_samples = np.sum([p['rewards'].shape[0] for p in data])
#     print(f'Number of samples collected: {num_samples}')
#     print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')


# with open('data/halfcheetah-poor.pkl', 'rb') as f:
#     data = pickle.load(f)
#     acc_r = []
#     for traj in data:
#         acc_r.append(sum(traj['rewards']))
#     print(acc_r)
# acc_r = enumerate(acc_r)
# acc_r.sorted(key = lambda x:x[1])
# print(acc_r[:10])
# max_id = acc_r.index(max(acc_r))
# print(data[1].keys())
# print(len(data[1]))

# path = 'data/door-poorQ.pkl'
# f = open(path, 'rb')
# data = pickle.load(f)
# print(data[99]['Q'])
