from modules.metrics import compute_scores

    
def evaluation(src_path, dst_path):
    # 数据加载
    with open(src_path, mode='r', encoding='utf-8') as f:
        reports = f.readlines()
    reports = [i.strip('\n') for i in reports if i != '\n']
    test_res = [i.split('：')[-1] for i in reports[::2]]
    test_gts = [i.split('：')[-1] for i in reports[1::2]]
    
    # 指标计算
    test_met = compute_scores({i: [gt] for i, gt in enumerate(test_gts)},
                              {i: [re] for i, re in enumerate(test_res)})
    result = ({'test_' + k: v for k, v in test_met.items()})
    
    # 文件保存
    res_str = ''
    for key, value in result.items():
        res_str += '\t{:15s}: {}\n'.format(str(key), value)
    print(res_str)
    with open(dst_path, mode='w', encoding='utf-8') as f:
        f.write(res_str)


if __name__ == '__main__':
    src_path = './results/mimic/mimic_kw_40/reports.txt'
    dst_path = src_path.replace('reports.txt', 'evaluating_indicator_with_cider.txt')
    evaluation(src_path, dst_path)