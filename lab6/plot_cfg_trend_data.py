import json
import matplotlib.pyplot as plt
import numpy as np

def plot_cfg_from_json():
    # 1. 讀取數據
    json_file = 'cfg_trend_data_ddim50.json'
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 找不到 {json_file}，請先執行 generate_cfg_trend_data.py！")
        return

    cfg_scales = data['cfg_scales']
    test_accs = data['test_accs']
    new_test_accs = data['new_test_accs']
    method = data['method']

    # 2. 設定圖表尺寸與高質感樣式
    plt.figure(figsize=(12, 7)) 

    # 畫出折線
    plt.plot(cfg_scales, test_accs, marker='o', markersize=8, linestyle='-', linewidth=2.5, 
             color='#1f77b4', label='test')
    plt.plot(cfg_scales, new_test_accs, marker='s', markersize=8, linestyle='-', linewidth=2.5, 
             color='#ff7f0e', label='new_test')

    # 設定標題與軸標籤
    title_str = f'Accuracy vs CFG Scale ({method})'
    plt.title(title_str, fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('CFG Scale', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    
    # X 軸顯示所有的 CFG 數值
    plt.xticks(cfg_scales)
    
    # Y 軸：最高到 1.1，設定刻度間距
    plt.ylim(0.0, 1.1) 
    plt.yticks(np.arange(0.0, 1.11, 0.1))

    # 設定圖例位置
    plt.legend(fontsize=12, loc='lower right', framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 讓排版緊湊後存檔
    plt.tight_layout()
    save_path = f'cfg_trend_plot_{method.lower()}50.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"✅ 繪圖完成！已儲存出高質感版本: {save_path}")

if __name__ == '__main__':
    plot_cfg_from_json()