from util import *
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--thread', default=1, type=int, help='the number for task threads')
    # parser.add_argument('--env', default='Pendulum-v0', type=str, help='open-ai gym environment')
    # parser.add_argument('--env', default='BipedalWalker-v3', type=str, help='open-ai gym environment')
    # parser.add_argument('--env', default='BipedalWalkerHardcore-v3', type=str, help='open-ai gym environment')
    # parser.add_argument('--env', default='Hopper-v2', type=str, help='open-ai gym environment')
    # parser.add_argument('--env', default='Ant-v2', type=str, help='open-ai gym environment')
    # parser.add_argument('--env', default='gym_kraby:HexapodBulletEnv-v0', type=str, help='open-ai gym environment')
    # parser.add_argument('--env', default='Walker2d-v2', type=str, help='open-ai gym environment')
    # parser.add_argument('--env', default='HalfCheetah-v2', type=str, help='open-ai gym environment')
    parser.add_argument('--env', default='pen-v0', type=str, help='open-ai gym environment')

    # parser.add_argument('--demonstration_path', default='data/door-expert-v0.pkl', type=str, help='the path of expert demonstration')
    parser.add_argument('--demonstration_path', default='data/relocate-expert-filterQ.pkl', type=str,
                        help='the path of expert demonstration')
    parser.add_argument('--demonstration_ratio', default=1, type=float)
    parser.add_argument('--demonstration_length', default=32, type=int)
    parser.add_argument('--bc_steps', default=100000, type=int)

    parser.add_argument('--debug', default=True, dest='debug', action='store_true')
    parser.add_argument('--seed', default=1, type=int, help='')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')

    parser.add_argument('--hidden1', default=256, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=512, type=int, help='hidden num of second fully connect layer')

    parser.add_argument('--rate', default=0.0001, type=float, help='learning rate')
    # parser.add_argument('--rate', default=0.00002, type=float, help='learning rate')

    parser.add_argument('--L2', default=0.001, type=float)
    # parser.add_argument('--L2', default=0.0001, type=float)

    parser.add_argument('--prate', default=0.001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=1000, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--bsize', default=256, type=int, help='minibatch size')

    # parser.add_argument('--rmsize', default=6000000, type=int, help='memory size')
    parser.add_argument('--rmsize', default=50000, type=int, help='memory size')

    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma')
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')

    parser.add_argument('--max_episode_length', default=2000, type=int, help='')

    parser.add_argument('--validate_episodes', default=5, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--validate_steps', default=10000, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--init_w', default=0.003, type=float, help='')

    parser.add_argument('--train_iter', default=1000000, type=int, help='train iters each timestep')

    parser.add_argument('--epsilon', default=800000, type=int, help='linear decay of exploration policy')
    # parser.add_argument('--l2norm', default=0.01, type=float, help='l2 weight decay')
    # parser.add_argument('--cuda', dest='cuda', TD3+TD3+TD3+ActionNoise='store_true')
    parser.add_argument('--num_interm', default=100, type=int, help='how many intermidate saves')
    parser.add_argument('--policy_delay', default=2, type=int)
    parser.add_argument('--noise_decay', default=1000000, type=int)

    args = parser.parse_args()
    return args