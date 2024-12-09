import json
import random

random.seed(123581)
data = json.load(open("../MaterialSet_allocation/train_sets.json", encoding='utf-8'))
single = []
simple = []
normal = []
for data_dict in data:
    set_num = len(data_dict['set'])
    if set_num == 1:
        single.append(data_dict)
    if set_num == 2:
        simple.append(data_dict)
    if set_num >= 10:
        print(data_dict['data_id'])

# #normal
# random_list = [i for i in range(len(data))]
# random_list = random.sample(random_list, int(len(data) * 0.1))
# for i in random_list:
#     normal.append(data[i])
#
# print(len(single))
# print(len(simple))
#
# simple_num = 40
# random_list = [i for i in range(len(simple))]
# random_list = random.sample(random_list, simple_num)
# simple_selected = []
# for i in random_list:
#     simple_selected.append(simple[i])
#
# with open("test_simple.json", "w") as f_simple:
#     json.dump(simple, f_simple)
# with open("test_single.json", "w") as f_single:
#     json.dump(single, f_single)
# with open("test_normal.json", "w") as f_normal:
#     json.dump(normal, f_normal)
