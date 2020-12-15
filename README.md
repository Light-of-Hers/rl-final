# 强化学习期末

[TOC]

## 小组成员

- 陈仁泽(1700012774)
- 谭钧尹(1800012956)
- 冯睿杰(1800013065)



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
      + `GameState`类：用于表示当前的对局状态，包含所有玩家的全局可见信息，以及自己部分可见信息。
        + `my_pid`：我的player id。
        + `my_hand`：我的手牌。
        + `players`：所有玩家的全局可见信息。
        + `history`：所有玩家的操作历史（包含摸牌操作和补花操作）。
        + `load_json(input_json)`：输入一个json对象，用其转换为对局状态。
        + `action_space()`：返回一个列表，列表的元素为元组，形如`(PLAY, Card0)`、`(PENG, Card1)`，与[botzone国标麻将的response](https://wiki.botzone.org.cn/index.php?title=Chinese-Standard-Mahjong)的格式类似。该列表表示当前所有的可行操作（可能存在重复的元素）。
    + `tactic/`：包含出牌策略。
      + `rand.py`：当前写的一个随机策略，仅从`action_space`中随机选取一个action返回。



## 构建bot

在项目根目录执行命令：

```python
python3 -m zipapp mj_bot
```

会在根目录获得可执行文件`mj_bot.pyz`，可直接用于在botzone上构建bot。



## 修改策略

在`tactic/`中添加新的文件、函数。一个出牌策略应该是一个函数，参数为一个`GameState`，返回一个action（类似`GameState.action_space`函数返回的列表的元素）。