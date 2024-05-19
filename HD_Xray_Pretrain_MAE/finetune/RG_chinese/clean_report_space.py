from compute_cider import evaluation

def clear(src_path, dst_path):
    # 数据加载
    with open(src_path, mode='r', encoding='utf-8') as f:
        reports = f.read()
    reports = reports.replace(' ', '')
    
    with open(dst_path, mode='w', encoding='utf-8') as f:
        f.write(reports)


if __name__ == '__main__':
    src_path = './results/mae_large_224_90_bs16_roberta/reports.txt'
    dst_path = src_path.replace('reports.txt', 'reports_wo_space.txt')
    clear(src_path, dst_path)
    
    print('Compute Soroces...')
    evaluation(src_path, src_path.replace('reports.txt', 'evaluating_indicator_with_cider_wo_space.txt'))