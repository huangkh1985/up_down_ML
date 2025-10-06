# 股票过滤配置说明

## 功能概述

本系统已添加自动过滤功能，可以排除以下类型的股票：
- **ST股票**: 包括ST、*ST等特别处理股票
- **次新股**: 上市时间不足指定天数的股票
- **数据不足股票**: 历史数据记录过少的股票

## 配置参数

### 1. ST股和次新股过滤 (在 `filter_stocks` 函数中)

可以在 `get_billboard_stocks()` 函数的返回前调整以下参数：

```python
filtered_stocks = filter_stocks(
    selected_stocks, 
    exclude_st=True,           # 是否排除ST股 (True/False)
    exclude_new_stock=True,    # 是否排除次新股 (True/False)
    new_stock_days=365         # 次新股定义：上市时间少于多少天
)
```

**参数说明:**
- `exclude_st`: 设为 `True` 排除ST股，`False` 保留ST股
- `exclude_new_stock`: 设为 `True` 排除次新股，`False` 保留次新股
- `new_stock_days`: 次新股判断标准，建议值：
  - `180` = 6个月（严格）
  - `365` = 1年（推荐）
  - `730` = 2年（宽松）

### 2. 数据记录数量要求 (在 `download_china_stock_enhanced_data` 函数中)

可以在 `main()` 函数中调整：

```python
stock_data = download_china_stock_enhanced_data(
    china_stocks, 
    start_date=analysis_start_date, 
    end_date=analysis_end_date,
    save_to_file=True,
    min_data_points=180  # 最少数据点要求
)
```

**参数说明:**
- `min_data_points`: 股票必须有的最少交易日数据，建议值：
  - `120` = 约6个月数据（宽松）
  - `180` = 约9个月数据（推荐）
  - `250` = 约1年数据（严格）

## 使用示例

### 示例1: 严格筛选（推荐用于模型训练）

```python
# 在 stock_data_downloader.py 的 get_billboard_stocks() 函数中
filtered_stocks = filter_stocks(
    selected_stocks, 
    exclude_st=True,           # 排除ST股
    exclude_new_stock=True,    # 排除次新股
    new_stock_days=365         # 上市不足1年的算次新股
)

# 在 main() 函数中
stock_data = download_china_stock_enhanced_data(
    china_stocks,
    min_data_points=250  # 要求至少1年数据
)
```

### 示例2: 宽松筛选（获取更多股票）

```python
# 在 stock_data_downloader.py 的 get_billboard_stocks() 函数中
filtered_stocks = filter_stocks(
    selected_stocks, 
    exclude_st=True,           # 排除ST股
    exclude_new_stock=True,    # 排除次新股
    new_stock_days=180         # 上市不足6个月的算次新股
)

# 在 main() 函数中
stock_data = download_china_stock_enhanced_data(
    china_stocks,
    min_data_points=120  # 要求至少6个月数据
)
```

### 示例3: 不过滤次新股（需要更多样本）

```python
filtered_stocks = filter_stocks(
    selected_stocks, 
    exclude_st=True,           # 只排除ST股
    exclude_new_stock=False,   # 不排除次新股
    new_stock_days=365
)
```

## 输出信息

程序运行时会显示详细的过滤信息：

```
开始过滤股票...
  原始股票数量: 30
  排除ST股: True
  排除次新股: True (上市<365天)

  ✓ 600519 (贵州茅台): 通过过滤
  ✗ 301357 (ST华信): ST股，已排除
  ✗ 688585 (泽璟制药): 次新股(上市245天)，已排除

过滤结果统计:
  通过过滤: 25 只
  排除ST股: 2 只
  排除次新股: 3 只
  获取信息失败: 0 只
```

## 修改位置

如需修改配置，请编辑以下文件：

**文件:** `stock_data_downloader.py`

**位置1 - ST股和次新股过滤** (约第204-209行):
```python
filtered_stocks = filter_stocks(
    selected_stocks, 
    exclude_st=True,           # 👈 在此修改
    exclude_new_stock=True,    # 👈 在此修改
    new_stock_days=365         # 👈 在此修改
)
```

**位置2 - 数据记录要求** (约第474行):
```python
stock_data = download_china_stock_enhanced_data(
    china_stocks,
    min_data_points=180  # 👈 在此修改
)
```

## 注意事项

1. **平衡样本数量**: 过滤太严格可能导致可用股票数量过少
2. **计算资源**: 更多股票意味着更长的计算时间
3. **模型质量**: ST股和次新股数据质量较差，建议排除
4. **数据完整性**: 确保股票有足够的历史数据用于特征提取

## 推荐配置

对于大多数分析场景，推荐使用当前默认配置：
- ✅ 排除ST股
- ✅ 排除上市不足1年的次新股
- ✅ 要求至少180天（9个月）的交易数据

这样可以在保证数据质量的同时，获得足够数量的高质量股票样本。
