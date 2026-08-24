import json
import matplotlib.pyplot as plt
import numpy as np

def plot_from_json():
    # 1. 讀取剛才算好的數據
    json_file = 'trend_data_ddpm.json'
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ 找不到 trend_data.json，請先執行 generate_trend_data.py！")
        return

    epochs = data['epochs']
    test_accs = data['test_accs']
    new_test_accs = data['new_test_accs']
    method = data['method']
    cfg = data['cfg_scale']

    # 2. 設定圖表尺寸與高質感樣式
    plt.figure(figsize=(14, 8)) 

    # 畫出折線 (加粗線條、加大標記)，修改為簡化的 label
    plt.plot(epochs, test_accs, marker='o', markersize=7, linestyle='-', linewidth=2.5, 
             color='#1f77b4', label='test')
    plt.plot(epochs, new_test_accs, marker='s', markersize=7, linestyle='-', linewidth=2.5, 
             color='#ff7f0e', label='new_test')


    # 設定標題與軸標籤
    title_str = f'Accuracy Trend across Training Epochs ({method}, CFG={cfg})'
    plt.title(title_str, fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)

    # 🌟 改善 X 軸的顯示：旋轉 45 度避免擁擠
    plt.xticks(epochs, rotation=45, ha='right') 
    
    # 🌟 改善 Y 軸：給頂部多一點留白空間 (最高到 1.1)，並設定刻度間距
    plt.ylim(0.0, 1.1) 
    plt.yticks(np.arange(0.0, 1.11, 0.1))

    # 設定圖例位置 (放在右下角，避免擋住高分區域)
    plt.legend(fontsize=12, loc='lower right', framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 讓排版緊湊後存檔
    plt.tight_layout()
    save_path = f'{json_file}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"✅ 繪圖完成！已儲存出高質感版本: {save_path}")

if __name__ == '__main__':
    plot_from_json()