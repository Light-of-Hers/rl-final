# 强化学习期末

[TOC]

## 小组成员与分工

- 陈仁泽(1700012774)：调研工作；基本框架的搭建；神经网络的部署训练。
- 谭钧尹(1800012956)：调研工作；rule-based bot的设计与实现。
- 冯睿杰(1800013065)：调研工作；CNN-based bot的设计与实现。



## 项目结构

+ `mj_bot/`
  + `__main__.py`：程序入口处，设置当前的策略(tactic)，处理和botzone的交互。
  + `mahjong/`：mahjong模块，代码主要放这里。
    + `common.py`：和游戏相关的一些玩意。
      + 一些常量字符串，比如`PASS`，`PLAY`，`PENG`，`CHI`等。
      + `Meld`类：用于表示一组明牌。
        + `type`：明牌的类型，分为`CHI`、`PENG`、`GANG`（顺子、刻子、杠子）。
        + `cards`：该明牌包含的牌，为字符串列表。
        + `src_pid`：吃、碰、直杠的对象的player id。
        + `src_card`：从吃、碰、直杠的对象处获得的牌。
      + `Player`类：用于表示玩家的全局可见信息。
        + `player_id`：字面意思。
        + `melds`：玩家的明牌。
        + `n_flowers`：玩家的花牌数。
        + `history`：玩家的操作历史（略去了摸牌操作和补花操作）。
        + `played`：玩家打出过的牌。
        + `n_left_cards`：牌墙剩下的牌数量。
      + `GameState`类：用于表示当前的对局状态，包含所有玩家的全局可见信息，以及自己部分可见信息。
        + `my_pid`：我的player id。
        + `my_hand`：我的手牌。
        + `players`：所有玩家的全局可见信息。
        + `history`：所有玩家的操作历史（包含摸牌操作和补花操作）。
        + `load_json(input_json)`：输入一个json对象，用其转换为对局状态。
        + `action_space()`：返回一个列表，列表的元素为元组，形如`(PLAY, Card0)`、`(PENG, Card1)`，与[botzone国标麻将的response](https://wiki.botzone.org.cn/index.php?title=Chinese-Standard-Mahjong)的格式类似（除了胡牌的动作，胡牌的动作为`(HU, n_fan)`，其中`n_fan`代表番数）。该列表表示当前所有的可行操作（可能存在重复的元素）。
    + `tactic/`：包含出牌策略。
      + `rand.py`：当前写的一个随机策略，仅从`action_space`中随机选取一个action返回（如果可以胡牌的话肯定会胡牌）。



## 项目使用

### 构建bot

运行根目录的文件`handin.py`：

```python
python3 handin.py
```

会在根目录获得可执行文件`mj_bot.pyz`，可直接用于在botzone上构建bot。



### 构建算番器

本地调试需要在本地构建算番器。在项目根目录执行：

```shell
cd 3rdparty/fan-calculator/Mahjong-GB-Python/
python3 setup.py install --user
```



### 修改策略

在`tactic/`中添加新的文件、函数。一个出牌策略应该是一个函数，参数为一个`GameState`，返回一个action（类似`GameState.action_space`函数返回的列表的元素）。



## 调研工作

### [*Method for Constructing Artificial Intelligence Player with Abstraction to Markov Decision Processes in Multiplayer Game of Mahjong*](https://arxiv.org/pdf/1904.07491.pdf)

+ 内容简介：
  + 形式化地构建了麻将游戏（该文中以日麻作为标准）中各个玩家的MDP模型。
  + 在构建的MDP的基础上推导出对局状态的估值函数（函数中的诸多参数需要借助统计学习手段获得）。
+ 对项目的帮助：
  + 本文更注重模型的构建与分析，对搜索策略的设计有一定启发，但暂无更多的帮助。



### [*Building a Computer Mahjong Player via Deep Convolutional Neural Networks*](https://arxiv.org/pdf/1906.02146v2.pdf)

[参见策略设计中的监督学习部分](#监督学习)



### [*A novel deep residual network-based incomplete information competition strategy for four-players Mahjong games*](https://www.researchgate.net/journal/Multimedia-Tools-and-Applications-1573-7721/publication/332881141_A_novel_deep_residual_network-based_incomplete_information_competition_strategy_for_four-players_Mahjong_games/links/5f75901692851c14bca41166/A-novel-deep-residual-network-based-incomplete-information-competition-strategy-for-four-players-Mahjong-games.pdf?_sg%5B0%5D=amkTcQ6o_rJdTlcjZFDc9XRLxgxsIpNoSYpLNb4RkEE9uO7OYOwKumZ8KQZM5acAX1KAlPLiq6d3HXASEBOJ2Q.Z-zRoSUklBStPXTIOBwmYPfvEZbdA0nBOQSDmixKiBpv_wLFE9T5zvZrzxMIS-hhu1Pz7_FclS0HEUs3mwsxTA&_sg%5B1%5D=sRDGRRFARzULWKLg4fNj4OJnce7Mow2qEonxyuQfhzMsOuSuRsthwTwbogxzu-yoaIPblOSqlG0-uCmef6yu7Q5x5_7MIi8lJrv5cnkua2qn.Z-zRoSUklBStPXTIOBwmYPfvEZbdA0nBOQSDmixKiBpv_wLFE9T5zvZrzxMIS-hhu1Pz7_FclS0HEUs3mwsxTA&_iepl=)



## 策略设计

### Rule-based



### 监督学习

[*Building a Computer Mahjong Player via Deep Convolutional Neural Networks*](https://arxiv.org/pdf/1906.02146v2.pdf)

监督学习希望使得agent的出牌和训练集的出牌尽量一致，提高agreement rate。

分别训练3个网络：出牌网络、碰网络和吃网络。杠的数据较少，直接采用规则。和牌番数足够就选择和。

#### 训练集

使用人类玩家的对局数据作为训练集。

##### 数据表示

- 数据是大小为63 x 34 × 4，是63个34 × 4的0/1矩阵。
- 矩阵的每个元素对应麻将的一张牌，34行分别对应麻将的34种花色，4列即每种牌4张
  - 1表示有该牌，0表示没有。
  - 某花色有1张牌，对应行就是1 0 0 0；有2张，对应行就是1 1 0 0；以此类推……
- 为了表示一个操作时的状态，共需要63个矩阵
  - 自己的手牌：1个
  - 4名玩家的明牌（吃、碰、杠）：4个
  - 4名玩家历史出过的牌：4个
  - 过去6次自己出牌时，以上的9个矩阵：9 × 6个

##### 标签

- 出牌网络：出什么牌，有34种可能性
- 碰网络：碰或不碰
- 吃网络：不吃，作为第1、2、3张吃，共4种可能性

#### 网络

- 3块，每块由1个卷积层、1个Batch normalization层和1个Dropout层组成，激活函数使用ReLU
  - 卷积层：卷积核大小为5 × 2，filters数量为100，不使用padding
  - Dropout 层：Dropout Rate设为0.5
- 之后再接上一个flatten层和一个300个神经元的全联接层
- 最后根据3个网络的不同特点使用34选1、2选1和4选1的输出层



### 强化学习

