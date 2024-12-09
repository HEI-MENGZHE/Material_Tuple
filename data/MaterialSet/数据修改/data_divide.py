import json
import random

random.seed(123581)

origin_data_path = "./train_sets.json"
data = json.load(open(origin_data_path, encoding='utf-8'))
text = data[0]["text"]
result = []
set_num = []
clock = 1
temp_dict = {"data_id": clock, "text": text, "set": []}
for mat_dict in data:
    mat_text = mat_dict["text"]
    if mat_text == text:
        temp_dict["set"].append(mat_dict["set"])
    else:
        clock += 1
        set_num.append(len(temp_dict["set"]))
        result.append(temp_dict)
        temp_dict = {"data_id": clock, "text": mat_text, "set": [mat_dict["set"]]}
        text = mat_text

# 输出文件
# write_path = "./try.json"
# with open(write_path, "w") as f_normal:
# json.dump(result, f_normal)

a = {}
for i in set_num:
    a[i] = set_num.count(i)
# {2: 93, 1: 63, 6: 11, 3: 66, 4: 31, 5: 11, 9: 2, 8: 1, 13: 1}

print(a)
print(sum(a.values()))
# 279

# 现在开始划分数据，设定1,2,3，>4四种数据集，test的条数都为n = 40, random 取9:1
single = []
double = []
triple = []
quadra = []
for data_dict in result:
    set_num = len(data_dict['set'])
    if set_num == 1:
        single.append(data_dict)
    if set_num == 2:
        double.append(data_dict)
    if set_num == 3:
        triple.append(data_dict)
    if set_num in [4,5,6]:
        quadra.append(data_dict)

# with open("quadra.json",'w') as fq:
#     json.dump(quadra,fq)

def divide_single(com_list):
    clock = 1
    result = []
    for com_set in com_list:
        for in_set in com_set["set"]:
            result.append({"data_id": clock, "text": com_set["text"], "set": in_set})
            clock += 1
    return result


# 先为single构造train, val, test
def gen_train_val_test(num, num_list, list1, list2, list3):
    test_num = 40
    index_list = [i for i in range(len(num_list))]  # test
    test_index = random.sample(index_list, test_num)[:30]
    test = []
    rest = []
    for index in test_index:
        test.append(num_list[index])

    for index_all in index_list:
        if index_all not in test_index:
            rest.append(num_list[index_all])

    train_val = list1 + list2 + list3 + rest
    random.shuffle(train_val)
    train = train_val[:int(0.9 * len(train_val))]
    val = train_val[int(0.9 * len(train_val)):]
    #在2以上的数据集中，dev和test中都有pred_set的字段，而train此时可以写入了
    # train 的 train_set 和 dev_set 一个在本文件中， 一个在allocation中， test_set也在本文件内
    # train_allocation的 train dev 都在allocation中
    train_all_path = "./MaterialSet_allocation/train_{}.json".format(num)
    dev_all_path = "./MaterialSet_allocation/dev_{}.json".format(num)
    for dev_dict in val:
        dev_dict['pred_set'] = dev_dict['set']

    with open(train_all_path, "w") as f4:
        json.dump(train, f4)
    with open(dev_all_path, "w") as f5:
        json.dump(val, f5)

    # test = divide_single(test) 在 num >= 2时，我们需要成捆的数据集
    train = divide_single(train)
    val = divide_single(val)

    test_path = "./test_{}.json".format(num)
    train_path = "./train_{}.json".format(num)
    val_path = "./dev_{}.json".format(num)
    with open(test_path, "w") as f1:
        json.dump(test, f1)
    with open(train_path, "w") as f2:
        json.dump(train, f2)
    with open(val_path, "w") as f3:
        json.dump(val, f3)

    return train, val, test

random.shuffle(result)
# 随机从四种数据集中抽取，1,2,3,4，还是直接抽取测试集要靠谱衣蛾

def gen_rest_test(num_list):
    index_list = [i for i in range(len(num_list))]
    test_index = random.sample(index_list, int(0.1*len(num_list)))
    test = []
    rest = []
    for index in test_index:

        test.append(num_list[index])
        # print(len(num_list[index]["set"]))

    for index_all in index_list:
        if index_all not in test_index:
            rest.append(num_list[index_all])
    print("single test:", len(test))
    return test, rest

def gen_random(list1, list2, list3, list4):
    test1, rest1 = gen_rest_test(list1)
    test2, rest2 = gen_rest_test(list2)
    test3, rest3 = gen_rest_test(list3)
    test4, rest4 = gen_rest_test(list4)
    print("1:{},2,:{},3:{},4:{}".format(len(test1),len(test2),len(test3),len(test4)))
    test_random = test1 + test2 + test3 + test4
    rest_random = rest4 + rest3 + rest2 + rest1
    print("len of test_random",len(test_random))
    random.shuffle(rest_random)
    train_random = rest_random[:int(0.9 * len(rest_random))]
    val_random = rest_random[int(0.9 * len(rest_random)):]

    train_random_all_path = "./MaterialSet_allocation/train_random.json"
    dev_random_all_path = "./MaterialSet_allocation/dev_random.json"

    for dev_dict in val_random:
        dev_dict['pred_set'] = dev_dict['set']

    with open(train_random_all_path, "w") as f8:
        json.dump(train_random, f8)
    with open(dev_random_all_path, "w") as f9:
        json.dump(val_random, f9)

    train_random = divide_single(train_random)
    val_random = divide_single(val_random)
    # test_random = divide_single(test_random)

    test_path = "./test_random.json"
    train_path = "./train_random.json"
    val_path = "./dev_random.json"
    # with open(test_path, "w") as f1:
    #     json.dump(test_random, f1)
    # with open(train_path, "w") as f2:
    #     json.dump(train_random, f2)


    # with open(val_path, "w") as f3:
    #     json.dump(val_random, f3)

    return train_random, val_random, test_random

# _, _, _ = gen_train_val_test(1, single, double, triple, quadra)
# _, _, _ = gen_train_val_test(2, double, single, triple, quadra)
# _, _, _ = gen_train_val_test(3, triple, single, double, quadra)
# _, _,  test = gen_train_val_test(4, quadra, single, double, triple)

_, _, _ = gen_random(single, double, triple, quadra)
print(6+18+18+3+16)