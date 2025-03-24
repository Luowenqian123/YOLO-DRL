import copy
import math

import numpy as np

class ENV():
    def __init__(self, UEs, MECs, k, lam):
        self.UEs = UEs
        self.MECs = MECs
        self.k = k
        self.a = 10

        q = np.full((k, 1), 0.)
        p = np.full((k, 1), 0.)

        q1 = np.linspace(0, 1, k).reshape((k, 1))
        q2 = np.linspace(0, 1, k).reshape((k, 1))
        q3 = np.full((k, 1), 0.)
        # q = np.array([])
        # p = np.array([])
        # q1 = np.array([])
        # 创建动作
        for j in range(k-1):
            q = np.append(q, q3, axis=0)
            b1 = np.full((k, 1), q2[j+1])
            p = np.append(p, b1, axis=0)
            q1 = np.append(q1, q2, axis=0)

        for i in range(MECs-1):

                b = np.linspace(0, 1, k).reshape((k, 1))
                for j in range(k):
                    a = np.full((k, 1), float(i + 1))
                    c = np.linspace(0, 1, k).reshape((k, 1))
                    b1 = np.full((k, 1), b[j])
                    q = np.append(q, a, axis=0)
                    p = np.append(p, b1, axis=0)
                    q1 = np.append(q1, c, axis=0)

        self.actions = np.hstack((q, p, q1))
        self.n_actions = len(self.actions)
        # self.n_features = 3 + MECs * 3
        self.n_features = 3 + MECs * 3
        self.discount = 0.01
        self.transmit = 10 ** (-1)
        self.noise = 10 ** (-13)
        self.channel = 4 * 10 ** (-13)

        # 基本参数
        # 频率
        self.Hz = 1
        self.kHz = 1000 * self.Hz
        self.mHz = 1000 * self.kHz
        self.GHz = 1000 * self.mHz
        self.nor = 10**(-7)
        self.nor1 = 10**9

        # 数据大小
        self.bit = 1
        self.B = 8 * self.bit
        self.KB = 1024 * self.B
        self.MB = 1024 * self.KB
        self.om = 0.5


        # self.task_cpu_cycle = np.random.randint(2 * 10**9, 3* 10**9)

        self.UE_f = np.random.randint(4.5 * self.GHz * self.nor, 6 * self.GHz * self.nor)     # UE的计算能力  最开始差5倍   1.5   2
        # self.MEC_f = np.random.randint(5 * self.GHz * self.nor, 7 * self.GHz * self.nor)  # MEC的计算能力
        # self.MEC_f = np.random.randint(7.5 * self.GHz * self.nor, 10 * self.GHz * self.nor)  # MEC的计算能力
        self.MEC_f = np.random.randint(15 * self.GHz * self.nor, 20 * self.GHz * self.nor)  # MEC的计算能力

        # self.UE_f = 500 * self.mHz     # UE的计算能力
        # self.MEC_f = np.random.randint(5.2 * self.GHz, 24.3 * self.GHz)  # MEC的计算能力
        self.tr_energy = 1      # 传输能耗
        # self.r = 40 * math.log2(1 + (16 * 10)) * self.MB * self.nor * 100 # 传输速率
        # self.r = math.log2(1 + (16 * 10)) * self.MB * self.nor   #40
        self.r = (math.log2(1 + (self.transmit * self.channel / self.noise))) * self.MB * self.nor * 5  #15
        # self.r = 800 # 传输速率
        self.ew, self.lw = 10**(-26), 10**(-28)# 能耗系数   3*10**(-28)
        # self.ew, self.lw = 0.3, 0.15 # 能耗系数
        self.et, self.lt = 1, 1
        self.local_core_max, self.local_core_min = 1.3 * self.UE_f, 0.7 * self.UE_f
        self.server_core_max, self.server_core_min = 1.3 * self.MEC_f, 0.7 * self.MEC_f
        self.uplink_max, self.uplink_min = 1.3 * self.r, 0.7 * self.r
        self.downlink_max, self.downlink_min = 1.3 * self.r, 0.7 * self.r
        self.lam = lam
        self.e = 0.1
        self.task_size2 = []
        self.task_size3 = []
        self.task_size4 = []
        self.task_cpu_cycle2 = []
        self.task_cpu_cycle3 = []
        self.task_cpu_cycle4 = []
        self.task_size5 = [1228800, 12288000000,12288000000, 18560, 73984, 115712, 295424, 131584, 33024, 90880, 296448, 0]
        # self.task_size5 = [1228800, 18560, 73984, 115712, 295424, 131584, 33024, 90880,296448, 0]
        for i in range(len(self.task_size5)):
            self.task_size2.append(self.task_size5[i] * self.nor)

        self.task_cpu_cycle5 = [0, 1.5, 1.5, 1.69, 3.62, 5.11, 6.06, 8.6, 9.87, 11.29, 14.4, 15.9]
        # self.task_cpu_cycle5 = [0, 1.69, 3.62, 5.11, 6.06, 8.6, 9.87, 11.29, 14.4, 15.9]
        for i in range(len(self.task_cpu_cycle5)):
            self.task_cpu_cycle2.append(self.task_cpu_cycle5[i] * self.nor1)

        self.task_size6 = [1228800, 12288000000, 12288000000, 55340, 9408, 21024, 269808, 254352, 927888, 876720, 1585488, 0]
        # self.task_size6 = [1228800, 55340, 9408, 21024, 269808, 254352, 927888, 876720,1585488, 0]
        for i in range(len(self.task_size6)):
            self.task_size3.append(self.task_size6[i] * self.nor)

        # self.task_cpu_cycle6 = [0, 0.46, 0.95, 1.66, 5.14, 5.96, 8.94, 9.64, 10.91, 11.8]
        self.task_cpu_cycle6 = [0, 0.2, 0.2, 0.46, 0.95, 1.66, 5.14, 5.96, 8.94, 9.64, 10.91, 11.8]
        for i in range(len(self.task_cpu_cycle5)):
            self.task_cpu_cycle3.append(self.task_cpu_cycle6[i] * self.nor1)

        # self.task_size7 = [1228800, 1760, 4800, 29184, 156928, 296448, 33024, 8320, 36992, 0]
        self.task_size7 = [1228800, 12288000000, 12288000000, 1760, 4800, 29184, 156928, 296448, 33024, 8320, 36992, 0]
        for i in range(len(self.task_size7)):
            self.task_size4.append(self.task_size7[i] * self.nor)
        self.task_cpu_cycle7 = [0, 0.1, 0.1, 0.37, 0.87, 1.49, 2.24, 2.72, 2.88, 3.19, 3.61, 4.6]
        # self.task_cpu_cycle7 = [0, 0.37, 0.87, 1.49, 2.24, 2.72, 2.88, 3.19, 3.61, 4.6]
        for i in range(len(self.task_cpu_cycle5)):
            self.task_cpu_cycle4.append(self.task_cpu_cycle7[i] * self.nor1)

        self.task_size5 = []
        self.task_size5.append(self.task_size2)
        self.task_size5.append(self.task_size3)
        self.task_size5.append(self.task_size4)

        self.task_cpu_cycle5 = []
        self.task_cpu_cycle5.append(self.task_cpu_cycle2)
        self.task_cpu_cycle5.append(self.task_cpu_cycle3)
        self.task_cpu_cycle5.append(self.task_cpu_cycle4)

    def reset(self):
        obs = []
        servers_cap = []
        normal_servers_cap = []
        new_cap = True
        UEs_ele2 = []
        for i in range(self.UEs):
            uplink, downlink = [], []
            normal_uplink, normal_downlink = [], []
            # np.random.seed(np.random.randint(1, 1000))
            # task_size = np.random.randint(2 * 10**8 * self.nor, 3 * 10**8 * self.nor) #   任务大小
            task_index = np.random.randint(0, 3)

            task_size = np.random.randint(10, 50)
            UEs_ele = np.random.randint(2500, 3000)  # 2500-3000
            # task_size = np.random.randint(1.5 * self.mHz, 2 * self.mHz) #   任务大小
            # self.task_size = self.task_size * self.task_cpu_cycle                     # 处理一个任务所需要的cpu频率
            # task_cpu_cycle = np.random.randint(2 * 10**9 * self.nor, 3 * 10**9 * self.nor)

            # task_cpu_cycle = np.random.randint(10**3, 10**5)
            # task_cpu_cycle = np.random.randint(10 ** 3, 2* 10 ** 3)
            # task_cpu_cycle = np.random.randint(5 * 10 ** 4, 10 ** 5)
            task_cpu_cycle = np.random.randint(5, 10)
            local_comp = np.random.randint(0.9 * self.UE_f, 1.1 * self.UE_f)  # UE的计算能力
            mec_cab = np.ones(self.MECs)
            for i in range(self.MECs):
                # up = np.random.randint(0.9 * self.r, 1.1 * self.r)
                # down = np.random.randint(0.9 * self.r, 1.1 * self.r)
                up = np.random.uniform(0.9 * self.r, 1.1 * self.r)
                normal_up = (up - self.uplink_min) / (self.uplink_max - self.uplink_min)
                down = np.random.uniform(0.9 * self.r, 1.1 * self.r)
                normal_down = (down - self.downlink_min) / (self.downlink_max - self.downlink_min)
                if new_cap:
                    cap = np.random.randint(0.9 * self.MEC_f, 1.1 * self.MEC_f)
                    # cap = np.random.randint(0.9 * self.MEC_f, 1.1 * self.MEC_f)  #MEC计算能力
                    cap = (cap - self.server_core_min) / (self.server_core_max - self.server_core_min)
                    # normal_cap = cap / 1.1 * self.MEC_f
                    # normal_servers_cap.append(normal_cap)
                    servers_cap.append(cap)

                uplink.append(up)
                downlink.append(down)
                normal_uplink.append(normal_up)
                normal_downlink.append(normal_down)
            normal_task_size = (task_size - 10) / (50 - 10)
            normal_task_cpu_cycle = (task_cpu_cycle - 5) / 5
            normal_local_comp = (local_comp - self.local_core_min) / (self.local_core_max - self.local_core_min)
            normal_ues_els = UEs_ele / 3000
            UEs_ele2.append(UEs_ele)

            observation = np.array([task_index, normal_local_comp, normal_ues_els])
            # observation = np.array([task_index, local_comp, normal_ues_els])
            observation = np.hstack((observation, servers_cap, normal_uplink, normal_downlink))
            # observation = np.hstack((observation, servers_cap, uplink, downlink))

            obs.append(observation)
            new_cap = False
        return obs, UEs_ele2

    def choose_action(self, prob):
        """
        根据概率选择动作
        :param env:
        :param prob:
        :return: [[target_server, percentage]]
        """
        action_choice = np.linspace(0, 1, self.k)
        actions = []
        for i in range(self.UEs):
            a = np.random.choice(a=(self.MECs * self.k * self.k), p=prob[i])  # 在数组p中从a个数字中以概率p选中一个
            target_server = int(a / (self.k * self.k))
            percen = action_choice[int(a / self.k) % self.k]
            percen1 = action_choice[a % self.k]
            action = [target_server, percen, percen1]
            actions.append(action)
        return actions

    def step(self, observation, actions_prob, ues_els1, is_prob=True, is_compared=True):
        if is_prob:
            actions = self.choose_action(actions_prob)
        else: actions = actions_prob
        new_cap = False
        obs_ = []
        rew, local, ran, mec = [], [], [], []
        dpg_times, local_times, ran_times, mec_times = [], [], [], []
        dpg_energys, local_energys, ran_energys, mec_energys = [], [], [], []
        total = []
        a, b, c, d = 0, 0, 0, 0
        mec_cab = np.ones(self.MECs)
        mec_cab1 = np.ones(self.MECs)
        mec_cab2 = np.ones(self.MECs)

        for i in range(self.UEs):
            if i == self.UEs - 1: new_cap = True
            # 提取信息
            task_index, local_comp1, normal_ues_els, servers_cap1, uplink1, downlink1= \
                observation[i][0], observation[i][1], observation[i][2], observation[i][3:3+self.MECs], observation[i][3+self.MECs:3+self.MECs*2], observation[i][3+self.MECs*2:3+self.MECs*3]
            # wait_local, wait_server = np.random.randint(0, 2), np.random.randint(0, 3)

            action = actions[i]
            target_server, percen, percen1 = int(action[0]), action[1], action[2]
            percen2 = int(percen * 11 + 0.001)   ########

            local_comp = (local_comp1 * (self.local_core_max - self.local_core_min) + self.local_core_min) / self.nor
            #local_comp = local_comp1 / self.nor
            uplink = uplink1 * (self.uplink_max - self.uplink_min) + self.uplink_min
            #uplink = uplink1
            downlink = downlink1 * (self.downlink_max - self.downlink_min) + self.downlink_min
            #downlink = downlink1
            servers_cap = (servers_cap1 * (self.server_core_max - self.server_core_min) + self.server_core_min) / self.nor
            #servers_cap = servers_cap1 / self.nor
            task_index = int(task_index + 0.1)
            ues_els = normal_ues_els * 3000
            # ues_els = normal_ues_els
            # print("task_index {}",task_index)
            # print("percen2 {}",percen2)


            # 计算奖励
            # 本地和服务器上都有
            tr_time = 0
            tr_energy = 0

            # comp_local_time = task_cpu_cycle * (1 - percen) / (local_comp)
            # comp_local_energy = self.lw * task_cpu_cycle * (1 - percen) * local_comp**2
            comp_local_time = self.task_cpu_cycle5[task_index][percen2] / local_comp
            comp_local_energy = self.lw * self.task_cpu_cycle5[task_index][percen2] * local_comp ** 2

            comp_local_time1 = 0
            comp_local_energy1 = 0
            comp_mec_time = 0

            # comp_mec_time = (percen * task_cpu_cycle) / servers_cap[target_server]


            # if mec_cab[target_server] == 0 or percen1 <= 0.01:
            #     comp_mec_time = task_cpu_cycle * percen / local_comp
            #     comp_mec_energy = self.lw * task_cpu_cycle * (percen) * local_comp**2
            # elif mec_cab[target_server] >= percen1 and percen1 >= 0.01:
            #     comp_mec_time = (percen * task_cpu_cycle) / (servers_cap[target_server] * percen1)
            #     comp_mec_energy = self.ew * percen * task_cpu_cycle * (servers_cap[target_server] * percen1) ** 2
            #     mec_cab[target_server] = mec_cab[target_server] - percen1
            # elif mec_cab[target_server] < percen1 and mec_cab[target_server] >= 0.01:
            #     comp_mec_time = (percen * task_cpu_cycle) / (servers_cap[target_server] * mec_cab[target_server])
            #     comp_mec_energy = self.ew * mec_cab[target_server] * task_cpu_cycle * (servers_cap[target_server] * percen1) ** 2
            #     mec_cab[target_server] = 0

            if mec_cab[target_server] <= 0.005 or percen1 <= 0.005:
                comp_local_time1 = (self.task_cpu_cycle5[task_index][9] - self.task_cpu_cycle5[task_index][percen2]) / local_comp
                comp_local_energy1 = self.lw * (self.task_cpu_cycle5[task_index][9] - self.task_cpu_cycle5[task_index][percen2]) * local_comp**2
            elif mec_cab[target_server] >= percen1 and percen1 >= 0.005:
                tr_time = (self.task_size5[task_index][percen2]) / uplink[target_server] + self.discount * (
                self.task_size5[task_index][percen2]) / downlink[target_server]
                tr_energy = (self.tr_energy * self.task_size5[task_index][percen2]) / uplink[
                    target_server] + self.discount * (self.tr_energy * self.task_size5[task_index][percen2]) / downlink[
                                target_server]
                comp_mec_time = (self.task_cpu_cycle5[task_index][9] - self.task_cpu_cycle5[task_index][percen2]) / (servers_cap[target_server] * percen1)
                comp_mec_energy = self.ew * (self.task_cpu_cycle5[task_index][9] - self.task_cpu_cycle5[task_index][percen2]) * (servers_cap[target_server] * percen1) ** 2
                mec_cab[target_server] = mec_cab[target_server] - percen1
            elif mec_cab[target_server] < percen1 and mec_cab[target_server] >= 0.005:
                tr_time = (self.task_size5[task_index][percen2]) / uplink[target_server] + self.discount * (
                self.task_size5[task_index][percen2]) / downlink[target_server]
                tr_energy = (self.tr_energy * self.task_size5[task_index][percen2]) / uplink[
                    target_server] + self.discount * (self.tr_energy * self.task_size5[task_index][percen2]) / downlink[
                                target_server]
                comp_mec_time = (self.task_cpu_cycle5[task_index][9] - self.task_cpu_cycle5[task_index][percen2]) / (servers_cap[target_server] * mec_cab[target_server])
                comp_mec_energy = self.ew * (self.task_cpu_cycle5[task_index][9] - self.task_cpu_cycle5[task_index][percen2]) * (servers_cap[target_server] * mec_cab[target_server]) ** 2
                mec_cab[target_server] = 0


            # comp_mec_energy = self.ew * percen * task_cpu_cycle * servers_cap[target_server]**2


            # comp_time = max(comp_local_time, comp_mec_time)

            comp_time = comp_local_time + comp_mec_time + comp_local_time1
            time_cost = (comp_time + tr_time) * self.et
            energy_cost = (tr_energy + comp_local_energy + comp_local_energy1) * self.e

            total_cost = self.lam * time_cost + (1 - self.lam) * energy_cost
            ues_els = ues_els - tr_energy - comp_local_energy - comp_local_energy1
            if ues_els > 0:
                ues_els = ues_els
            else:
                ues_els = 0

            # reward = -total_cost

            # 全本地
            # local_only_time = task_cpu_cycle/(local_comp) * self.et
            # local_only_energy = self.lw * task_cpu_cycle * local_comp**2 * self.e

            local_only_time = self.task_cpu_cycle5[task_index][9] / (local_comp) * self.et
            local_only_energy = self.lw * self.task_cpu_cycle5[task_index][9] * local_comp ** 2 * self.e
            # local_only_energy = task_size * local_comp
            local_only = self.lam * local_only_time + (1 - self.lam) * local_only_energy
            # print("task_cpu_cycle:", task_cpu_cycle)
            # print("local_comp", local_comp)
            # print("local_only_time:", local_only_time)
            # print("local_only_energy:", local_only_energy)
            # print("local_only:", local_only)
            e_a = (local_only_energy - energy_cost) / local_only_energy
            t_a = (local_only_time - (comp_time + tr_time)) / local_only_time
            e_re = ues_els / 3000
            f_m = self.om * e_re * t_a + (1 - self.om * e_re) * e_a
            u_qoe = math.pow(self.a, f_m) - 1

            # 全边缘

            # 有问题这里MEC资源可能存在重复使用
            # for m in range(self.MECs):
            #     if mec_cab1[m] >= (self.MECs / self.UEs):
            #         mec_only_tr_time = task_size / uplink[m] + self.discount * task_size / downlink[
            #             m]
            #         mec_only_tr_energy = self.tr_energy * task_size / uplink[
            #             m] + self.discount * self.tr_energy * task_size / downlink[m]
            #         mec_only_comp_time = task_cpu_cycle / (servers_cap[m] * self.MECs / self.UEs)
            #         mec_only_comp_energy = self.ew * task_cpu_cycle * (
            #                     servers_cap[m] * self.MECs / self.UEs) ** 2
            #         break
            #     elif mec_cab1[m] <= (self.MECs / self.UEs) and mec_cab1[m] != 0:
            #         mec_only_tr_time = task_size / uplink[m] + self.discount * task_size / downlink[
            #             m]
            #         mec_only_tr_energy = self.tr_energy * task_size / uplink[
            #             m] + self.discount * self.tr_energy * task_size / downlink[m]
            #         mec_only_comp_time = task_cpu_cycle / (servers_cap[m] * mec_cab1[m])
            #         mec_only_comp_energy = self.ew * task_cpu_cycle * (
            #                 servers_cap[m] * mec_cab1[m]) ** 2
            #         break

            for m in range(self.MECs):
                if mec_cab1[m] >= (self.MECs / self.UEs):
                    mec_only_tr_time = self.task_size5[task_index][0] / uplink[m] + self.discount * self.task_size5[task_index][0] / downlink[
                        m]
                    mec_only_tr_energy = self.tr_energy * self.task_size5[task_index][0] / uplink[
                        m] + self.discount * self.tr_energy * self.task_size5[task_index][0] / downlink[m]
                    mec_only_comp_time = (self.task_cpu_cycle5[task_index][9] - self.task_cpu_cycle5[task_index][0]) / (servers_cap[m] * self.MECs / self.UEs)
                    mec_only_comp_energy = self.ew * (self.task_cpu_cycle5[task_index][9] - self.task_cpu_cycle5[task_index][0]) * (
                            servers_cap[m] * self.MECs / self.UEs) ** 2
                    break
                elif mec_cab1[m] <= (self.MECs / self.UEs) and mec_cab1[m] != 0:
                    mec_only_tr_time = self.task_size5[task_index][0] / uplink[m] + self.discount * self.task_size5[task_index][0] / downlink[
                        m]
                    mec_only_tr_energy = self.tr_energy * self.task_size5[task_index][0] / uplink[
                        m] + self.discount * self.tr_energy * self.task_size5[task_index][0] / downlink[m]
                    mec_only_comp_time = (self.task_cpu_cycle5[task_index][9] - self.task_cpu_cycle5[task_index][0]) / (servers_cap[m] * mec_cab1[m])
                    mec_only_comp_energy = self.ew * (self.task_cpu_cycle5[task_index][9] - self.task_cpu_cycle5[task_index][0]) * (
                            servers_cap[m] * mec_cab1[m]) ** 2
                    break

            # mec_only_tr_time = task_size / uplink[target_server] + self.discount * task_size / downlink[target_server]
            # mec_only_tr_energy = self.tr_energy * task_size / uplink[
            #     target_server] + self.discount * self.tr_energy * task_size / downlink[target_server]

            # print("mec_only_tr_time:", mec_only_tr_time)
            # print("mec_only_tr_energy:", mec_only_tr_energy)


            # mec_only_comp_time = task_cpu_cycle / (servers_cap[target_server] * self.MECs / self.UEs)
            # mec_only_comp_energy = self.ew * task_cpu_cycle * (servers_cap[target_server] * self.MECs / self.UEs) **2
            # print("mec_only_comp_time:", mec_only_comp_time)
            # print("mec_only_comp_energy:", mec_only_comp_energy)

            mec_only_time_cost = (mec_only_tr_time + mec_only_comp_time) * self.et
            mec_only_energy_cost = (mec_only_tr_energy) * self.e

            mec_only = self.lam * mec_only_time_cost + (1 - self.lam) * mec_only_energy_cost
            # print("mec_only_time_cost:", mec_only_time_cost)
            # print("mec_only_energy_cost:", mec_only_energy_cost)
            # print("----------------------------:", servers_cap[target_server])


            # 随机卸载
            percen_ran = np.random.uniform()    # 随机卸载比例
            mec_ran = np.random.randint(self.MECs)  # 随机选择一个服务器进行卸载
            percen_ran1 = np.random.uniform()
            percen_ran2 = int(percen_ran * 10)

            random_tr_time = self.task_size5[task_index][percen_ran2] / uplink[mec_ran] + (self.discount * self.task_size5[task_index][percen_ran2]) / downlink[mec_ran]
            random_tr_energy = (self.tr_energy * self.task_size5[task_index][percen_ran2]) / uplink[mec_ran] + self.discount * (self.tr_energy * self.task_size5[task_index][percen_ran2]) / downlink[mec_ran]

            random_comp_local_time = self.task_cpu_cycle5[task_index][percen_ran2] / local_comp
            random_comp_local_energy = self.lw * self.task_cpu_cycle5[task_index][percen_ran2] * local_comp**2
            # random_comp_local_energy = (1 - percen_ran) * task_size * local_comp

            # random_comp_mec_time = percen_ran * task_cpu_cycle / servers_cap[mec_ran]
            # random_comp_mec_energy = self.ew * percen_ran * task_cpu_cycle * servers_cap[mec_ran]**2
            # # random_comp_mec_energy = percen_ran * task_size * servers_cap[mec_ran]
            random_comp_mec_energy1 = 0
            random_comp_mec_time = 0
            random_comp_mec_time1 = 0

            if mec_cab2[mec_ran] <= 0.005 or percen_ran1 <= 0.005:
                random_comp_mec_time1 = (self.task_cpu_cycle5[task_index][9] - self.task_cpu_cycle5[task_index][percen_ran2]) / local_comp
                random_comp_mec_energy1 = self.lw * (self.task_cpu_cycle5[task_index][9] - self.task_cpu_cycle5[task_index][percen_ran2]) * local_comp ** 2
            elif mec_cab2[mec_ran] >= percen_ran1 and percen_ran1 >= 0.005:
                random_comp_mec_time = (self.task_cpu_cycle5[task_index][9] - self.task_cpu_cycle5[task_index][percen_ran2]) / (servers_cap[mec_ran] * percen_ran1)
                random_comp_mec_energy = self.ew * (self.task_cpu_cycle5[task_index][9] - self.task_cpu_cycle5[task_index][percen_ran2]) * (servers_cap[mec_ran] * percen_ran1) ** 2
                mec_cab2[mec_ran] = mec_cab2[mec_ran] - percen_ran1
            elif mec_cab2[mec_ran] < percen_ran1 and mec_cab2[mec_ran] >= 0.005:
                random_comp_mec_time = self.ew * (self.task_cpu_cycle5[task_index][9] - self.task_cpu_cycle5[task_index][percen_ran2]) / (servers_cap[mec_ran] * mec_cab2[mec_ran])
                random_comp_mec_energy = self.ew * (self.task_cpu_cycle5[task_index][9] - self.task_cpu_cycle5[task_index][percen_ran2]) * (
                            servers_cap[mec_ran] * mec_cab2[mec_ran]) ** 2
                mec_cab2[mec_ran] = 0

            # random_comp_time = max(random_comp_local_time, random_comp_mec_time)
            random_comp_time = random_comp_local_time + random_comp_mec_time + random_comp_mec_time1

            random_time_cost = (random_comp_time + random_tr_time) * self.et
            random_energy_cost = (random_tr_energy + random_comp_local_energy + random_comp_mec_energy1) * self.e
            # ues_els1 = ues_els1 - random_tr_energy + random_comp_local_energy + random_comp_mec_energy1

            # random_total = self.lam * random_time_cost + (1 - self.lam) * random_energy_cost

            e_a1 = (local_only_energy - random_energy_cost) / local_only_energy
            t_a1 = (local_only_time - random_time_cost) / local_only_time
            ues_els1[i] = ues_els1[i] - tr_energy - comp_local_energy - comp_local_energy1
            if ues_els1[i] > 0:
                ues_els1[i] = ues_els1[i]
            else:
                ues_els1[i] = 0
            e_re1 = ues_els1[i] / 3000
            f_m1 = self.om * e_re1 * t_a1 + (1 - self.om * e_re1) * e_a1
            random_total = math.pow(self.a, f_m1) - 1
            if random_total < -10:
                print(random_total)
            random_total_cost2 = random_energy_cost

            # if total_cost < random_total or total_cost < mec_only or total_cost < local_only:
            #     reward = -total_cost
            # else:
            #     print("惩罚")
            #     reward = -1999

            # reward = -total_cost
            reward = u_qoe

            # a += total_cost
            # b += mec_only
            # c += local_only
            # d += random_total

            # 得到下一个observation
            x = np.random.uniform()
            y = 0.5
            if (x > y):
                local_comp = min(local_comp * self.nor + np.random.randint(0, 0.2 * self.UE_f), self.local_core_max)
                for j in range(self.MECs):
                    cap = (min(servers_cap[j] * self.nor + np.random.randint(0, 0.3 * self.UE_f), self.server_core_max) - self.server_core_min) / (self.server_core_max -self.server_core_min)
                    #cap = min(servers_cap[j] * self.nor + np.random.randint(0, 0.3 * self.UE_f), self.server_core_max)
                    # MEC容量保持一致
                    if new_cap:
                        for x in range(self.UEs):
                            observation[x][2 + j] = cap
                    downlink[j] = (min(downlink[j] + np.random.uniform(0, 0.2 * self.r), self.downlink_max) - self.downlink_min) / (self.downlink_max - self.downlink_min)
                    #downlink[j] = min(downlink[j] + np.random.randint(0, 0.2 * self.r), self.downlink_max)
                    #uplink[j] = min(uplink[j] + np.random.randint(0, 0.2 * self.r), self.uplink_max)

                    uplink[j] = (min(uplink[j] + np.random.uniform(0, 0.2 * self.r), self.uplink_max) - self.uplink_min) / (self.uplink_max - self.uplink_min)
            else:
                local_comp = max(local_comp * self.nor + np.random.randint(-0.2 * self.UE_f, 0), self.local_core_min)
                for j in range(self.MECs):
                    # MEC容量保持一致
                    if new_cap:
                        cap = (max(servers_cap[j] * self.nor - np.random.randint(0, 0.3 * self.UE_f), self.server_core_min) - self.server_core_min) / (self.server_core_max - self.server_core_min)
                        #cap = max(servers_cap[j] * self.nor - np.random.randint(0, 0.3 * self.UE_f),
                                  #self.server_core_min)
                        for x in range(self.UEs):
                            observation[x][2 + j] = cap
                    downlink[j] = (max(downlink[j] - np.random.uniform(0, 0.2 * self.r), self.downlink_min) - self.downlink_min) / (self.downlink_max - self.downlink_min)
                    #downlink[j] = max(downlink[j] - np.random.randint(0, 0.2 * self.r), self.downlink_min)
                    uplink[j] = (max(uplink[j] - np.random.uniform(0, 0.2 * self.r), self.uplink_min) - self.uplink_min) / (self.uplink_max - self.uplink_min)
                    #uplink[j] = max(uplink[j] - np.random.randint(0, 0.2 * self.r), self.uplink_min)

            # task_size = np.random.randint(10, 50)
            task_size = np.random.randint(10, 50)
            task_index = np.random.randint(0, 3)
            # task_cpu_cycle = np.random.randint(10**3, 10**5)  # 处理任务所需要的CPU频率
            task_cpu_cycle = np.random.randint(5, 10)

            normal_task_size = (task_size - 10) / (50 - 10)
            normal_task_cpu_cycle = (task_cpu_cycle - 5) / 5
            normal_local_comp = (local_comp - self.local_core_min) / (self.local_core_max - self.local_core_min)
            ues_els = ues_els / 3000
            # ues_els = ues_els

            observation_ = np.array([task_index, normal_local_comp, ues_els])
            #observation_ = np.array([task_index, local_comp, ues_els])
            observation_ = np.hstack((observation_, servers_cap1, uplink, downlink))
            obs_.append(observation_)

            rew.append(reward)
            local.append(local_only)
            mec.append(mec_only)
            ran.append(random_total)


            dpg_times.append(time_cost)
            local_times.append(local_only_time)
            mec_times.append(mec_only_time_cost)
            ran_times.append(random_time_cost)

            dpg_energys.append(energy_cost)
            local_energys.append(local_only_energy)
            mec_energys.append(mec_only_energy_cost)
            ran_energys.append(random_energy_cost)

            total.append(total_cost)

        # if (a - b > 10 * self.UEs) or (a - c > 10 * self.UEs) or (a - d > 10 * self.UEs):
        #     print("惩罚")
        #     # print(a ,b, c, d)
        #     for i in range(self.UEs):
        #         rew[i] = -999
        # else:
        #     pass

        if is_compared:
            return obs_, rew, local, mec, ran, dpg_times, local_times, mec_times, ran_times, dpg_energys, local_energys, mec_energys, ran_energys, total, ues_els1
        else:
            return obs_, rew, dpg_times, dpg_energys , total , ues_els1
            # return obs_, total

