from ..common import *


# shanten difficulty; the chance to get key cards would affect shanten
st_diff = [10, 1.1, 1, 0.9, 0.8]
fan_distribute = [0.08, 0.105, 0.106, 0.236, 0.041, 0.01, 0.017]
path_visit = []


def basic_shanten_01(tiles):
    ans = 2
    cards = tiles[:]
    for card in cards:
        if cards.count(card) >= 2:
            return 0
        if REMAIN_CARDS[card] > 0:
            shanten = st_diff[REMAIN_CARDS[card]]
            if shanten < ans:
                ans = shanten
    return ans


def basic_shanten(tiles, pair_need, mianzi_need, path=[], priority=2, debug_flag=0, shunzi_lower=1, shunzi_upper=9):
    if debug_flag:
        print("tiles:", tiles, "path:", path, "pair:",
              pair_need, "mianzi:", mianzi_need)
    ans = 10
    if path in path_visit:
        return 10
    path_visit.append(path)
    if mianzi_need == 0:
        if pair_need == 0:
            return 0
        return basic_shanten_01(tiles)
    # 面子
    cards = tiles[:]
    cards.sort()
    cards_set = set(cards)
    priority_flag = priority
    if priority_flag == 2:
        priority_flag = 1
        # 刻子
        for card in cards_set:
            if cards.count(card) == 3:
                priority_flag = 2
                visit_cards = [card, card, card]
                [cards.remove(card) for i in range(3)]
                shanten = basic_shanten(
                    cards, pair_need, mianzi_need - 1, path + visit_cards, 2, debug_flag=debug_flag)
                if shanten < ans:
                    ans = shanten
                [cards.append(card) for i in range(3)]
        # 顺子
        for i in 'WBT':
            for j in range(shunzi_lower, shunzi_upper - 1):
                if i + str(j) in cards and i + str(j + 1) in cards and i + str(j + 2) in cards:
                    priority_flag = 2
                    visit_cards = [i + str(j), i + str(j + 1), i + str(j + 2)]
                    [cards.remove(card) for card in visit_cards]
                    shanten = basic_shanten(
                        cards, pair_need, mianzi_need - 1, path + visit_cards, 2, debug_flag=debug_flag)
                    if shanten < ans:
                        ans = shanten
                    [cards.append(card) for card in visit_cards]
        if priority_flag == 2:
            return ans
    if priority_flag == 1:
        priority_flag = 0
        # 搭子,来到这里是因为kezi_need > 0,所以要找搭子
        for i in 'WBT':
            for j in range(shunzi_lower, shunzi_upper - 1):
                if i + str(j) in cards and i + str(j + 2) in cards:
                    if REMAIN_CARDS[i + str(j + 1)] == 0:
                        continue
                    priority_flag = 1
                    visit_cards = [i + str(j), i + str(j + 2)]
                    [cards.remove(card) for card in visit_cards]
                    shanten = st_diff[REMAIN_CARDS[i + str(j + 1)]] + basic_shanten(
                        cards, pair_need, mianzi_need - 1, path + visit_cards, 1, debug_flag=debug_flag)
                    if shanten < ans:
                        ans = shanten
                    [cards.append(card) for card in visit_cards]
            for j in range(shunzi_lower, shunzi_upper):
                if i + str(j) in cards and i + str(j + 1) in cards:
                    x, y = 10, 10
                    if j > shunzi_lower:
                        x = st_diff[REMAIN_CARDS[i + str(j - 1)]]
                    elif j < shunzi_upper - 1:
                        y = st_diff[REMAIN_CARDS[i + str(j + 2)]]
                    if x == 10 and y == 10:
                        continue
                    diff = min(x, y)
                    if x != 10 and y != 10:
                        diff -= 0.1
                    priority_flag = 1
                    visit_cards = [i + str(j), i + str(j + 1)]
                    [cards.remove(card) for card in visit_cards]
                    shanten = diff + basic_shanten(
                        cards, pair_need, mianzi_need - 1, path + visit_cards, 1, debug_flag=debug_flag)
                    if shanten < ans:
                        ans = shanten
                    [cards.append(card) for card in visit_cards]
        for card in cards_set:
            if cards.count(card) == 2:
                if REMAIN_CARDS[card] == 0:
                    continue
                priority_flag = 1
                visit_cards = [card, card]
                [cards.remove(card) for card in visit_cards]
                shanten = st_diff[REMAIN_CARDS[card]] + basic_shanten(
                    cards, pair_need, mianzi_need - 1, path + visit_cards, 1, debug_flag=debug_flag)
                if shanten < ans:
                    ans = shanten
                [cards.append(card) for card in visit_cards]
        if priority_flag == 1:
            return ans
    # 剩下的就是啥也没有
    if debug_flag:
        print(mianzi_need * 2 + pair_need)
    return mianzi_need * 2 + pair_need


def connection_bonus(cards):
    bonus = 0
    pools = {'W': [], 'B': [], 'T': [], 'F': [], 'J': []}
    for card in cards:
        pools[card[0]].append(int(card[1]))
    for pool in pools:
        for card in set(pools[pool]):
            cnt = pools[pool].count(card)
            if cnt == 1:
                if pool == 'F' or pool == 'J':
                    bonus -= 0.013
            if cnt == 2:
                bonus += 0.03
            elif cnt == 3:
                bonus += 0.05
        if pool != 'F' and pool != 'J':
            for card in set(pools[pool]):
                if pools[pool].count(card) == 1:
                    if card == 1:
                        if 2 not in pools[pool] and 3 not in pools[pool]:
                            bonus -= 0.012
                    elif card == 9:
                        if 8 not in pools[pool] and 7 not in pools[pool]:
                            bonus -= 0.012
                    elif card == 2:
                        if 1 not in pools[pool] and 3 not in pools[pool] and 4 not in pools[pool]:
                            bonus -= 0.011
                    elif card == 8:
                        if 9 not in pools[pool] and 7 not in pools[pool] and 6 not in pools[pool]:
                            bonus -= 0.011
                    else:
                        if card - 1 not in pools[pool] and card + 1 not in pools[pool] \
                                and card - 2 not in pools[pool] and card + 2 not in pools[pool]:
                            bonus -= 0.01
    return bonus


def target_table_shunzi(shunzi):
    target_table = [[] for i in range(7)]  # shunzi_type
    type_s = ['WBT', 'WTB', 'BWT', 'BTW', 'TBW', 'TWB']
    for i in range(3):  # 清龙
        target = []
        for q in range(1, 10, 3):
            target.append([type_s[0][i] + str(q), type_s[0][i] +
                           str(q + 1), type_s[0][i] + str(q + 2)])
        flag = 1
        for shun in shunzi:
            if shun not in target:
                flag = 0
                break
        if flag == 1:
            target = target[0] + target[1] + target[2]
            target_table[0].append(target)

    for i in range(6):  # 花龙
        target = []
        for j in range(3):
            target.append([type_s[i][j] + str(3 * j + 1), type_s[i]
                           [j] + str(3 * j + 2), type_s[i][j] + str(3 * j + 3)])
        flag = 1
        for shun in shunzi:
            if shun not in target:
                flag = 0
                break
        if flag == 1:
            target = target[0] + target[1] + target[2]
            target_table[1].append(target)

    for i in range(1, 8):  # 三色三同顺
        target = []
        for j in range(3):
            target.append([type_s[0][j] + str(i), type_s[0][j] +
                           str(i + 1), type_s[0][j] + str(i + 2)])
        flag = 1
        for shun in shunzi:
            if shun not in target:
                flag = 0
                break
        if flag == 1:
            target = target[0] + target[1] + target[2]
            target_table[2].append(target)

    for i in range(30):  # 三色三步高
        target = []
        k1, k2 = (i // 6) + 1, i % 6
        for t in range(3):
            target.append([type_s[k2][t] + str(k1 + t), type_s[k2][t] +
                           str(k1 + t + 1), type_s[k2][t] + str(k1 + t + 2)])
        flag = 1
        for shun in shunzi:
            if shun not in target:
                flag = 0
                break
        if flag == 1:
            target = target[0] + target[1] + target[2]
            target_table[3].append(target)

    for i in range(15):  # 一色三步高
        target = []
        k1, k2 = i // 5, (i % 5) + 1
        for t in range(3):
            target.append([type_s[0][k1] + str(k2 + t), type_s[0][k1] +
                           str(k2 + t + 1), type_s[0][k1] + str(k2 + t + 2)])
        flag = 1
        for shun in shunzi:
            if shun not in target:
                flag = 0
                break
        if flag == 1:
            target = target[0] + target[1] + target[2]
            target_table[4].append(target)

    for i in range(3):  # 一色三同顺
        for q in range(1, 8):
            target = []
            for k in range(3):
                target.append([type_s[0][i] + str(q), type_s[0]
                               [i] + str(q + 1), type_s[0][i] + str(q + 2)])
            flag = 1
            for shun in shunzi:
                if shun not in target:
                    flag = 0
                    break
            if flag == 1:
                target = target[0] + target[1] + target[2]
                target_table[5].append(target)

    if len(shunzi) <= 1:  # 组合龙
        for i in range(6):
            target = []
            for t in range(3):
                target.extend([type_s[i][t] + str(t + 1), type_s[i]
                               [t] + str(t + 4), type_s[i][t] + str(t + 7)])
            if shunzi:
                for shun in shunzi:
                    target.extend(shun)
            target_table[6].append(target)
    return target_table


def search_shunzi_shanten_shunzi_branch(hands, mianzi_need, pair_need, shunzi=[]):
    global path_visit
    hands_fulu = hands[:]
    for shun in shunzi:
        hands_fulu += shun
    target_table = target_table_shunzi(shunzi)
    shantens = []
    ans = 10
    final_target = []
    final_need = 0
    final_remain = []
    final_bonus = 0
    final_b_st = 0
    hands_fulu.sort()
    for i in range(7):
        for target in target_table[i]:
            need = []
            tmp_hands = hands_fulu[:]
            for target_card in target:
                if target_card in tmp_hands:
                    tmp_hands.remove(target_card)
                else:
                    need += [target_card]
            remain = tmp_hands[:]
            shanten_flag = 0
            for card in need:
                if REMAIN_CARDS[card] == 0:
                    shanten_flag = 1
                    break
            if shanten_flag == 1:
                continue
            diff = 0
            for card in need:
                diff += st_diff[REMAIN_CARDS[card]]
            path_visit = []
            # print("target:", target)
            # print("remain:", remain, "need：", need)
            b_st = basic_shanten(remain, pair_need, mianzi_need)
            bonus = connection_bonus(remain)
            shanten = diff + b_st - bonus
            # print("shanten:%.2f, b_st:%.2f" % (shanten, b_st))
            if shanten > 3:
                shanten -= fan_distribute[i]
            shantens.append(shanten)
            if ans > shanten:
                ans = shanten
                final_bonus = bonus
                final_b_st = b_st
                final_target = target
                final_need = need
                final_remain = remain
    # print(final_target, final_need, final_remain, final_bonus, final_b_st, ans)
    return shantens


# 顺子类：清龙，花龙，三色三同顺，三色三步高，一色三步高，一色三同顺，组合龙 (fan_type:0/1/2/3/4/5)
# 大数据番型分布：[8 10.5 10.6 23.6 4.1 1](小番型一律按1算)
def search_least_shanten_shunzi(hands, melds):
    kezi = 0
    shunzi = []
    shantens = []
    ans = 10
    for meld in melds:
        if meld.type == PENG or meld.type == GANG:
            kezi += 1
        elif meld.type == CHI:
            shunzi.append(meld.cards)
    if kezi >= 2:
        return ans  # 已经有两个刻子，不能往顺子系的牌型做了
    elif kezi == 1:
        shantens += search_shunzi_shanten_shunzi_branch(hands, 0, 1, shunzi)
    elif kezi == 0:
        if not shunzi:
            shantens += search_shunzi_shanten_shunzi_branch(hands, 1, 1)
        elif len(shunzi) == 1:
            shantens += search_shunzi_shanten_shunzi_branch(
                hands, 1, 1, shunzi)
            shantens += search_shunzi_shanten_shunzi_branch(hands, 0, 1)
        elif len(shunzi) == 2:
            shantens += search_shunzi_shanten_shunzi_branch(
                hands, 1, 1, shunzi)
            shantens += search_shunzi_shanten_shunzi_branch(
                hands, 0, 1, [shunzi[0]])
            shantens += search_shunzi_shanten_shunzi_branch(
                hands, 0, 1, [shunzi[1]])
        elif len(shunzi) == 3:
            shantens += search_shunzi_shanten_shunzi_branch(
                hands, 1, 1, shunzi)
            shantens += search_shunzi_shanten_shunzi_branch(
                hands, 0, 1, [shunzi[0], shunzi[1]])
            shantens += search_shunzi_shanten_shunzi_branch(
                hands, 0, 1, [shunzi[0], shunzi[2]])
            shantens += search_shunzi_shanten_shunzi_branch(
                hands, 0, 1, [shunzi[1], shunzi[2]])
    for shanten in shantens:
        if shanten < ans:
            ans = shanten
    backup = 0
    for shanten in shantens:
        if shanten - ans <= 1:
            backup += 1  # backup用来选出具有最多的[次优目标牌型]的状态
    # print("ans: %.3f  backup: %d" % (ans, backup))
    return ans - backup / 100


# 混一色
def search_least_shanten_mixed1color(hands, melds):
    global path_visit
    ans = 10
    main_colors = ['F', 'J', 'F']
    for meld in melds:
        t = meld.cards[0][0]
        if t not in main_colors:
            if main_colors[2] == 'F':
                main_colors[2] = t
            else:
                return ans
    pools = {'W': [], 'B': [], 'T': [], 'F': [], 'J': []}
    for card in hands:
        pools[card[0]].append(int(card[1]))
    if main_colors[2] == 'F':
        main_cnt = -1
        for pool in pools:
            if pool != 'F' and pool != 'J':
                if len(pools[pool]) > main_cnt:
                    main_cnt = len(pools[pool])
                    main_colors[2] = pool
    minor = 13 - 3 * \
        len(melds) - len(pools[main_colors[2]]) - \
        len(pools['F']) - len(pools['J'])
    if minor > 3:
        return 2 * minor
    # minor<=3:
    remain = hands[:]
    for card in hands:
        if card[0] not in main_colors:
            remain.remove(card)

    path_visit = []
    shanten = 0.8 * basic_shanten(remain, 1, 4 -
                                  len(melds), debug_flag=0) + 0.2 * minor - 0.06
    return shanten


# 大于五
def search_least_shanten_morethan5(hands, melds):
    global path_visit
    ans = 10
    for meld in melds:
        for card in meld.cards:
            if card[0] == 'F' or card[0] == 'J' or int(card[1]) < 6:
                return ans
    remain = hands[:]
    minor = 0
    for card in hands:
        if card[0] == 'F' or card[0] == 'J' or int(card[1]) < 6:
            remain.remove(card)
            minor += 1
    if minor > 3:
        return 2 * minor
    path_visit = []
    shanten = 0.8 * basic_shanten(remain, 1, 4 -
                                  len(melds), debug_flag=0) + 0.2 * minor - 0.014
    return shanten


# 小于五
def search_least_shanten_lessthan5(hands, melds):
    global path_visit
    ans = 10
    for meld in melds:
        for card in meld.cards:
            if card[0] == 'F' or card[0] == 'J' or int(card[1]) > 4:
                return ans
    remain = hands[:]
    minor = 0
    for card in hands:
        if card[0] == 'F' or card[0] == 'J' or int(card[1]) > 4:
            remain.remove(card)
            minor += 1
    if minor > 3:
        return 2 * minor
    path_visit = []
    shanten = 0.8 * basic_shanten(remain, 1, 4 -
                                  len(melds), debug_flag=0) + 0.2 * minor - 0.013
    return shanten


# 五门齐_附件
def shanten_kezi(pools, pool):
    shanten = 10
    if not pools:
        return shanten
    for card in pools:
        tmp = (3 - pools.count(card)) * st_diff[REMAIN_CARDS[pool + str(card)]]
        if tmp < shanten:
            shanten = tmp
    return shanten


# 五门齐_附件
def shanten_shunzi(pools, pool):
    shanten = 10
    if not pools:
        return shanten
    for i in range(1, 8):
        tmp = 0
        for card in range(i, i + 3):
            tmp += (1 - int(card in pools)) * \
                st_diff[REMAIN_CARDS[pool + str(card)]]
        if tmp < shanten:
            shanten = tmp
    return shanten


# 五门齐
def search_least_shanten_alltype(hands, melds):
    ans = 10
    shantens = {'W': 3, 'B': 3, 'T': 3, 'F': 3, 'J': 3}
    for meld in melds:
        shantens[meld.cards[0][0]] -= 3
        if shantens[meld.cards[0][0]] < 0:
            return ans

    pools = {'W': [], 'B': [], 'T': [], 'F': [], 'J': []}
    need = [3, 3, 3, 3, 3]
    for card in hands:
        pools[card[0]].append(int(card[1]))
    for pool in pools:
        tmp = shanten_kezi(pools[pool], pool)
        if tmp < shantens[pool]:
            shantens[pool] = tmp
        if pool == 'W' or pool == 'B' or pool == 'T':
            tmp = shanten_shunzi(pools[pool], pool)
            if tmp < shantens[pool]:
                shantens[pool] = tmp
    shanten = shantens['W'] + shantens['B'] + shantens['T'] + \
        max(0, shantens['F'] + shantens['J'] - 1) - 0.107
    return shanten


# 全求人
def search_least_shanten_allbyothers(hands, melds):
    shanten1 = 9 - 2 * len(melds)
    global path_visit
    # 将上听数近似为搭子数
    remain = hands[:]
    path_visit = []
    shanten2 = basic_shanten(remain, 1, 4 - len(melds),
                             debug_flag=0) - connection_bonus(remain)
    return (shanten1 + shanten2) / 2


st_diff2 = [2, 1.6, 1.4, 1.2, 1]


# 七对
def search_least_shanten_7pairs(hands, melds):
    if melds:
        return 10
    shanten = 0
    pools = {'W': [], 'B': [], 'T': [], 'F': [], 'J': []}
    for card in hands:
        pools[card[0]].append(int(card[1]))
    for pool in pools:
        for card in set(pools[pool]):
            cnt = pools[pool].count(card)
            if cnt == 3 or cnt == 1:
                shanten += (st_diff2[REMAIN_CARDS[pool + str(card)]])
    return shanten / 2


# 十三幺
def search_least_shanten_13yao(hands, melds):
    if melds:
        return 10
    shanten = 10
    yao_card = ['F1', 'F2', 'F3', 'F4', 'J1', 'J2',
                'J3', 'B1', 'B9', 'T1', 'T9', 'W1', 'W9']
    target_table = [['F1', 'F2', 'F3', 'F4', 'J1', 'J2', 'J3',
                     'B1', 'B9', 'T1', 'T9', 'W1', 'W9'] for i in range(12)]
    for i in range(12):
        target_table[i].append(yao_card[i])
    for target in target_table:
        need = []
        tmp_hands = hands[:]
        for target_card in target:
            if target_card in tmp_hands:
                tmp_hands.remove(target_card)
            else:
                need += [target_card]
        # remain = tmp_hands[:]
        shanten_flag = 0
        for card in need:
            if REMAIN_CARDS[card] == 0:
                shanten_flag = 1
                break
        if shanten_flag == 1:
            continue
        diff = 0
        for card in need:
            diff += st_diff[REMAIN_CARDS[card]]
        if shanten > diff:
            shanten = diff
    return shanten


# 全不靠
def search_least_shanten_0acquaintance(hands, melds):
    if melds:
        return 10
    type_s = ['WBT', 'WTB', 'BWT', 'BTW', 'TBW', 'TWB']
    shanten = 10
    target_table = [['F1', 'F2', 'F3', 'F4', 'J1', 'J2', 'J3']
                    for i in range(6)]
    for i in range(6):  # 组合龙
        for t in range(3):
            target_table[i].extend(
                [type_s[i][t] + str(t + 1), type_s[i][t] + str(t + 4), type_s[i][t] + str(t + 7)])
    for target in target_table:
        need = []
        tmp_hands = hands[:]
        for target_card in target:
            if target_card in tmp_hands:
                tmp_hands.remove(target_card)
            else:
                need += [target_card]
        # remain = tmp_hands[:]
        shanten_flag = 0
        for card in need:
            if REMAIN_CARDS[card] == 0:
                shanten_flag = 1
                break
        if shanten_flag == 1:
            continue
        diffs = []
        diff = 0
        for card in need:
            diffs.append(st_diff[REMAIN_CARDS[card]])
        diffs.sort()
        for i in range(len(diffs) - 2):
            diff += diffs[i]
        if shanten > diff:
            shanten = diff
    return shanten


# 碰碰和
def search_least_shanten_pengpenghu(hands, melds):
    shanten = 0
    kezi_need = 4
    dazi_need = 0
    pair_need = 1
    tmp_hands = hands[:]
    for meld in melds:
        if meld.type == CHI:
            return 10
        kezi_need -= 1
    for card_id in range(len(tmp_hands) - 1, -1, -1):
        if card_id < len(tmp_hands) - 2 and tmp_hands[card_id] == tmp_hands[card_id + 1] and tmp_hands[card_id] == \
                tmp_hands[card_id + 2]:
            [tmp_hands.remove(tmp_hands[card_id]) for i in range(3)]
            card_id -= 1
            kezi_need -= 1
    if kezi_need == 0:
        if REMAIN_CARDS[tmp_hands[0]] == 0:
            return 2
        return st_diff[REMAIN_CARDS[tmp_hands[0]]]
    # 刻子不够，找搭子
    dazi_need = kezi_need
    for card_id in range(len(tmp_hands) - 1, -1, -1):
        if card_id < len(tmp_hands) - 1 and tmp_hands[card_id] == tmp_hands[card_id + 1]:
            if REMAIN_CARDS[tmp_hands[card_id]] == 0:
                [tmp_hands.remove(tmp_hands[card_id]) for i in range(2)]
                if pair_need == 1:  # 如果已经有雀头了那只能对这个对子视而不见了(完全没用，直接删掉)
                    pair_need -= 1
            else:
                dazi_need -= 1
                shanten += st_diff[REMAIN_CARDS[tmp_hands[card_id]]]
                [tmp_hands.remove(tmp_hands[card_id]) for i in range(2)]
                if dazi_need == 0:
                    break
    if dazi_need == 0:
        return shanten
    diffs = []
    for card in tmp_hands:
        diffs.append(st_diff[REMAIN_CARDS[card]])
    diffs.sort()
    for diff in diffs:
        if dazi_need > 0:
            shanten += diff * 2
            dazi_need -= 1
        elif pair_need > 0:
            shanten += diff
            pair_need -= 1
        else:
            break
    if dazi_need > 0:
        shanten += 2 * dazi_need
    if pair_need > 0:
        shanten += 2
    return shanten


# 三暗刻
def search_least_shanten_3anke(hands, melds):
    global path_visit
    kezi_need = 3
    for meld in melds:
        if meld.type == CHI:
            return 10
        kezi_need -= 1
    shantens = []
    tmp_hands = hands[:]
    for card in set(tmp_hands):
        shantens.append([(3 - tmp_hands.count(card)) * st_diff[REMAIN_CARDS[card]],
                         [card for i in range(tmp_hands.count(card))]])
    shantens.sort(key=lambda x: x[0])
    shanten = 0
    for i in range(len(shantens)):
        if kezi_need == 0:
            break
        shanten += shantens[i][0]
        [tmp_hands.remove(card) for card in shantens[i][1]]
        kezi_need -= 1
    path_visit = []
    shanten += basic_shanten(tmp_hands, 1, 1)
    return shanten


def value_function2(game_state, act, debug=0):
    if act[0] == HU or act[0] == BUGANG:
        return 10000
    hands = game_state.my_hand[:]
    melds = game_state.players[game_state.my_pid].melds[:]
    value = 10
    if act[0] == PLAY:
        hands.remove(act[1])
    elif act[0] == PENG:
        card0 = (game_state.history[-1])[-1]
        [hands.remove(card0) for i in range(2)]
        melds.append(Meld(PENG, [card0, card0, card0]))
        hands.remove(act[1])
    elif act[0] == CHI:
        value -= 0.2
        cards = [act[1][0] + str(int(act[1][1]) - 1),
                 act[1], act[1][0] + str(int(act[1][1]) + 1)]
        hands.append((game_state.history[-1])[-1])
        [hands.remove(card) for card in cards]
        melds.append(Meld(CHI, cards))
        hands.remove(act[2])
    elif act[0] == GANG:
        value += 1.5
        if game_state.history[-1][-1] == 'DRAW':
            melds.append(Meld(GANG, [act[1] for i in range(4)]))
            [hands.remove(act[1]) for i in range(4)]
        else:
            card0 = (game_state.history[-1])[-1]
            melds.append(Meld(GANG, [card0 for i in range(4)]))
            [hands.remove(card0) for i in range(3)]
    hands.sort()
    st1 = search_least_shanten_shunzi(hands, melds)
    st2 = search_least_shanten_alltype(hands, melds)
    st3 = search_least_shanten_pengpenghu(hands, melds)
    st4 = search_least_shanten_mixed1color(hands, melds)
    st5 = search_least_shanten_allbyothers(hands, melds)
    st6 = search_least_shanten_7pairs(hands, melds)
    st7 = search_least_shanten_13yao(hands, melds)
    st8 = search_least_shanten_0acquaintance(hands, melds)
    st9 = search_least_shanten_morethan5(hands, melds)
    st10 = search_least_shanten_lessthan5(hands, melds)
    st11 = search_least_shanten_3anke(hands, melds)
    if debug:
        print("顺子:%.2f 五门齐:%.2f 碰碰和:%.2f 混一色:%.2f 全求人:%.2f 七对:%.2f 十三幺:%.2f 全不靠:%.2f 大于五:%.2f "
              "小于五:%.2f 三暗刻:%.2f" % (st1, st2, st3, st4, st5, st6, st7, st8, st9, st10, st11))
    st = [st1, st2, st3, st4, st5, st6, st7, st8, st9, st10, st11]
    st.sort()
    if st[0] > 2:
        shanten = st[0] + st[1] * 0.01 + st[2] * 0.005
        # print(st[0],st[1],st[2])
    else:
        shanten = st[0]
    if debug:
        print("动作: %s  上听数: %.3f\n" % (act, 10 - value + shanten))
    return value - shanten


class RuleBasedTactic:
    def loading(self, game_state: GameState, pid: int, action: tuple):
        pass

    def __call__(self, game_state: GameState):
        space = game_state.action_space()
        if next((act for act in space if act[0] == HU), None) is not None:
            return (HU,)
        max_value = -1000
        max_act = None
        game_state.remain_cards_calc()
        for act in space:
            value = value_function2(game_state, act, debug=0)
            if value > max_value:
                max_value = value
                max_act = act
        return max_act
