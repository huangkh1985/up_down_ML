"""
matplotlib中文显示配置
统一配置matplotlib的中文字体
"""

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties


def configure_chinese_font():
    """
    配置matplotlib以正确显示中文
    """
    # 尝试多个中文字体
    font_candidates = [
        'Microsoft YaHei',  # 微软雅黑（Windows推荐）
        'SimHei',           # 黑体
        'SimSun',           # 宋体
        'KaiTi',            # 楷体
        'FangSong',         # 仿宋
        'STSong',           # 华文宋体（Mac）
        'Arial Unicode MS', # Mac备选
        'sans-serif'        # 默认
    ]
    
    # 设置字体列表
    plt.rcParams['font.sans-serif'] = font_candidates
    
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设置默认字体大小
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    
    # 设置figure参数
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    
    return font_candidates[0]


def get_chinese_font():
    """
    获取可用的中文字体
    
    返回:
    font_name: 字体名称
    """
    from matplotlib.font_manager import FontManager
    import matplotlib.font_manager as fm
    
    fm_fonts = fm.fontManager.ttflist
    
    # Windows常见中文字体
    preferred_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
    
    for preferred in preferred_fonts:
        for font in fm_fonts:
            try:
                if preferred in font.name:
                    return font.name
            except:
                pass
    
    return 'sans-serif'


# 自动配置
_font_name = configure_chinese_font()
print(f"[matplotlib] 使用字体: {_font_name}")
