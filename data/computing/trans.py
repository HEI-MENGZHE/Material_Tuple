import json

def split_json(input_file, output_file):
    """
    将输入的json文件中set列表中的字典拆分成独立的json数据，输出到指定文件

    Args:
        input_file: 输入json文件路径
        output_file: 输出json文件路径
    """
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        data = json.load(f_in)
        new_data = []
        data_id = 1
        for item in data:
            for sub_item in item["set"]:
                new_data.append({
                    "data_id": data_id,
                    "text": item["text"],
                    "set": sub_item
                })
                data_id += 1
        json.dump(new_data, f_out, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    input_file = 'train_computing_backup.json'
    output_file = 'train_computing.json'
    split_json(input_file, output_file)
    print(f"拆分后的数据已写入 {output_file}")