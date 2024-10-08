import gym
import numpy as np
from matplotlib import pyplot as plt
from IPython import display
from gym import spaces

class GymHelper:
    def __init__(self, env, figsize=(3, 3)):
        self.env = env  # 初始化Gym环境
        self.figsize = figsize  # 初始化绘图窗口大小

        plt.figure(figsize=figsize)  # 创建绘图窗口
        plt.title(self.env.spec.id if hasattr(env.spec, 'id') else '')  # 标题设为环境名
        self.img = plt.imshow(env.render())  # 在绘图窗口中显示初始图像

    def render(self, title=None):
        image_data = self.env.render()  # 获取当前环境图像渲染数据

        self.img.set_data(image_data)  # 更新绘图窗口中的图像数据
        display.display(plt.gcf())  # 刷新显示
        display.clear_output(wait=True)  # 有新图片时再清除绘图窗口原有图像
        if title:  # 如果有标题就显示标题
            plt.title(title)

class EVs_Env(gym.Env):
    # 构造函数，参数为node
    def __init__(self, EVs, env_info):

        super(EVs_Env, self).__init__()

        # 环境信息
        self.data = env_info['data']
        self.distance = env_info['distance']
        self.speed = env_info['speed']
        self.roads = env_info['roads']
        self.node_road = env_info['node_road']
        self.node_num = len(self.data)
        self.EVs = EVs

        # 定义动作空间,动作空间为[0,1,2,...,49],选择动作n表示到n号节点
        self.action_space = spaces.Discrete(self.node_num)
        # # 定义观测空间
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, 1), dtype=np.uint8)

        self.start = self.EVs['start']
        self.end = self.EVs['end']
        self.init_power = self.EVs['init_power']
        self.max_power = self.EVs['max_power']
        self.dead_line = self.EVs['dead_line']
        self.consumption = self.EVs['consumption'] / 100

        # 剩余时间
        self.remain_time = self.dead_line
        # 当前电量
        self.current_power = self.init_power
        # 当前处在几号节点
        self.current = self.start

    # 环境重置函数
    def reset(self):
        self.start = self.EVs['start']
        self.end = self.EVs['end']
        self.init_power = self.EVs['init_power']
        self.max_power = self.EVs['max_power']
        self.dead_line = self.EVs['dead_line']
        self.consumption = self.EVs['consumption'] / 100    # 注意单位

        self.current = self.start

        self.remain_time = self.dead_line
        self.current_power = self.init_power
        return self._get_state(), {}

    # 动作执行函数    函数返回[observation, reward, terminated, info]
    def step(self, action):

        # 如果选择去的下一个节点是当前节点的话，重新选择
        while action == self.current:
            # print('重新选择动作')
            action = self.sample()

        is_charge = self.roads[self.current][action]     # 是否充电路段
        distance = self.distance[self.current][action]   # 距离
        speed = self.speed[self.current][action]         # 行驶速度
        time_consuming = distance/speed                  # 行驶耗时
        self.remain_time -= time_consuming               # 剩余时间
        consumption = distance * self.consumption        # 行驶消耗

        if is_charge:
            charge_power = 100 * time_consuming     # 充电功率为100Kw
            if charge_power <= consumption:
                self.current_power -= (consumption - charge_power)
            else:
                self.current_power = min(self.current_power + (charge_power - consumption), self.max_power)
        else:
            self.current_power -= consumption


        # 执行动作后，如果剩余时间<0或者电量<0则表示在半路就终止了
        if self.remain_time < 0:
            return self._get_state(), 0, True, {}

        if self.current_power < 0:
            return self._get_state(), 0, True, {}

        # 否则更新当前位置为执行动作后的位置
        self.current = action

        if self.current == self.end:
            return self._get_state(), self.current_power, True, {}
        else:
            return self._get_state(), 0, False, {}

    # 在动作空间随机选择一个动作
    def sample(self):
        return self.action_space.sample()

    def _get_state(self):
        return [self.current, self.current_power, self.remain_time]
