from tensorboardX import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator

# 'data/iter_reward_sum'

ea1=event_accumulator.EventAccumulator(r'C:\PYTHON\python_project\pytorch-TD3\output\Walker2d\Walker2d-with-noise\tensorboardx_log\events.out.tfevents.1628392339.LAPTOP-N6MT5K2P')
ea1.Reload()

# ea2=event_accumulator.EventAccumulator(r'C:\PYTHON\python_project\pytorch-TD3\output\Walker2d-BC+FineTune-72\tensorboardx_log\events.out.tfevents.1628403493.LAPTOP-N6MT5K2P')
# ea2.Reload()

val1=ea1.scalars.Items('data/iter_reward_sum')
# val2=ea2.scalars.Items('data/iter_reward_sum')

writer1 = SummaryWriter(log_dir='CompPic/CompPic-Walker2d/TD3+ActionNoise')
# writer2 = SummaryWriter(log_dir='CompPic\curv2')

i = 0
for i in val1:
    if i.step >= 600000:
        break
    writer1.add_scalar('data/iter_reward_sum', i.value, i.step)

# for i in val2:
#     writer2.add_scalar('iter_reward_sum', i.value, i.step)