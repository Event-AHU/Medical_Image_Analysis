import json
import re

def find_word_indices(sen, target_word):
    target_words = re.sub('(?<! )(?=[/,-,:,.,!?()])|(?<=[/,-,:,.,!?()])(?! )', r' ', target_word).lower().split()
    start_index = -1
    end_index = -1
    for i, word in enumerate(sen):
        if word == target_words[0] and (start_index == -1 or end_index == -1):
            if sen[i:i + len(target_words)] == target_words:
                start_index = i
                end_index = i + len(target_words) - 1
    return start_index, end_index


def get_ner_dict(sen, res_dict_id):
    # 只保留类型为 "anatomy" 或 "disorder" 的实体
    entities_dict = {}
    for entity, entity_type in res_dict_id.items():
        start_index, end_index = find_word_indices(sen, entity)
        if start_index != -1 and end_index != -1 and entity_type in ["anatomy", "disorder"]:
        # if start_index != -1 and end_index != -1 and entity_type in ["anatomy"]:
        # if start_index != -1 and end_index != -1 and entity_type in ["anatomy", "disorder", "devices", "concept", "procedures"]:
        # if start_index != -1 and end_index != -1 and entity_type in ["anatomy", "disorder", "devices"]:
            # 只保留类型为 "anatomy" 或 "disorder" 的实体
            entities_dict[entity] = entity_type
    
    return entities_dict


def get_sentence_list(ori_report):
    # 去掉换行符
    ori_report = str(ori_report).replace('\n', '')
    # 用句号分割句子
    sentence_list = str(ori_report).split('.')
    return_sentence_list = []

    idx = 0
    while idx < len(sentence_list):
        sentence_idx = sentence_list[idx]
        # 检查当前句子是否以数字结尾以及下一个句子是否以数字开头
        if (idx + 1 < len(sentence_list) and 
            re.search(r'\d$', sentence_idx) and 
            re.match(r'^\d', sentence_list[idx + 1])):
            # 将当前句子与下一个句子合并
            return_sentence_list.append(sentence_idx + '.' + sentence_list[idx + 1])
            idx += 1  # 跳过下一个句子
        else:
            return_sentence_list.append(sentence_idx)
        idx += 1

    return return_sentence_list


def preprocess_report(report_text, res_dict_id):
    sentence_list = get_sentence_list(report_text)
    
    final_list = []
    for sen_idx, sentence in enumerate(sentence_list):
        sen = re.sub('(?<! )(?=[/,-,:,.,!?()])|(?<=[/,-,:,.,!?()])(?! )', r' ', sentence).lower().split()
        if len(sen) < 2:
            continue  # 跳过太短的句子

        temp_dict = {}
        temp_dict["doc_key"] = f"report_{sen_idx}"
        temp_dict["sentences"] = [sen]
        temp_dict["entities"] = get_ner_dict(sen, res_dict_id)  # 返回类型为 "anatomy" 或 "disorder" 的实体
        final_list.append(temp_dict)

    return final_list

# 把 preprocess_report 的结果拿过来，统一合并所有句子的 entities：
def merge_entities(results):
    merged = {}
    for item in results:
        for ent, ent_type in item["entities"].items():
            # 如果已经存在，就不覆盖；也可以选择覆盖，取最后一个
            if ent not in merged:
                merged[ent] = ent_type
    return merged


# 示例执行
if __name__ == '__main__':
    report_text = "unchanged position of the left upper extremity PICC line. Again seen are surgical clips projecting over the right hemithorax. The cardiomediastinal silhouette is stable in appearance. Increased stranding opacities are noted in the left retrocardiac region. Subtle stranding opacities in the right upper lung zone are unchanged.. There are no pleural or significant bony abnormalities. Absence of the right breast shadow compatible with prior mastectomy."

    # 加载 res_dict_id 字典
    res_dict_file = '/data/qiaoyuhan/hhhh/Zretrival/res_dict_aliases.json'  # 修改为实际的JSON文件路径
    with open(res_dict_file, 'r') as file:
        res_dict_id = json.load(file)

    # 处理报告文本
    result = preprocess_report(report_text, res_dict_id)
    
    merged_entities = merge_entities(result)
    print(json.dumps(merged_entities, indent=4))


    # # 打印输出结果
    # for item in result:
    #     print(json.dumps(item, indent=4))
