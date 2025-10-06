"""
股票数据下载模块
负责从efinance获取股票数据，包括K线、资金流等信息
"""

import efinance as ef
import pandas as pd
import numpy as np
import sys
sys.path.append('.')
from utils.technical_indicators import add_technical_indicators
from datetime import datetime, timedelta

def filter_stocks(stock_codes, exclude_st=True, exclude_new_stock=True, new_stock_days=365):
    """
    过滤股票列表，排除ST股和次新股
    
    参数:
    stock_codes: 股票代码列表
    exclude_st: 是否排除ST股
    exclude_new_stock: 是否排除次新股
    new_stock_days: 次新股定义天数（上市时间小于该天数的股票）
    
    返回:
    过滤后的股票代码列表
    """
    print(f"\n开始过滤股票...")
    print(f"  原始股票数量: {len(stock_codes)}")
    print(f"  排除ST股: {exclude_st}")
    print(f"  排除次新股: {exclude_new_stock} (上市<{new_stock_days}天)")
    
    filtered_stocks = []
    excluded_st = []
    excluded_new = []
    excluded_error = []
    
    for stock_code in stock_codes:
        try:
            # 获取股票基本信息
            try:
                base_info = ef.stock.get_base_info(stock_code)
            except:
                # 如果获取失败，尝试获取实时行情（包含股票名称）
                try:
                    quote = ef.stock.get_realtime_quotes([stock_code])
                    if quote is not None and not quote.empty:
                        stock_name = quote.iloc[0]['股票名称'] if '股票名称' in quote.columns else ''
                        base_info = {'股票名称': stock_name, '上市日期': None}
                    else:
                        base_info = None
                except:
                    base_info = None
            
            if base_info is None:
                print(f"  ⚠ {stock_code}: 无法获取股票信息，跳过")
                excluded_error.append(stock_code)
                continue
            
            # 检查是否是ST股
            stock_name = base_info.get('股票名称', '')
            if exclude_st and stock_name:
                if 'ST' in stock_name or 'st' in stock_name or '*ST' in stock_name:
                    print(f"  ✗ {stock_code} ({stock_name}): ST股，已排除")
                    excluded_st.append((stock_code, stock_name))
                    continue
            
            # 检查是否是次新股
            if exclude_new_stock:
                listing_date = base_info.get('上市日期', None)
                if listing_date:
                    try:
                        # 处理各种日期格式
                        if isinstance(listing_date, str):
                            if len(listing_date) == 8:  # YYYYMMDD
                                listing_date = datetime.strptime(listing_date, '%Y%m%d')
                            elif '-' in listing_date:  # YYYY-MM-DD
                                listing_date = datetime.strptime(listing_date, '%Y-%m-%d')
                        elif isinstance(listing_date, pd.Timestamp):
                            listing_date = listing_date.to_pydatetime()
                        
                        days_since_listing = (datetime.now() - listing_date).days
                        
                        if days_since_listing < new_stock_days:
                            print(f"  ✗ {stock_code} ({stock_name}): 次新股(上市{days_since_listing}天)，已排除")
                            excluded_new.append((stock_code, stock_name, days_since_listing))
                            continue
                    except Exception as e:
                        print(f"  ⚠ {stock_code}: 上市日期解析失败({listing_date})，保留")
            
            # 通过所有过滤条件
            filtered_stocks.append(stock_code)
            print(f"  ✓ {stock_code} ({stock_name}): 通过过滤")
            
        except Exception as e:
            print(f"  ⚠ {stock_code}: 处理异常({e})，跳过")
            excluded_error.append(stock_code)
            continue
    
    # 打印过滤统计
    print(f"\n过滤结果统计:")
    print(f"  通过过滤: {len(filtered_stocks)} 只")
    print(f"  排除ST股: {len(excluded_st)} 只")
    print(f"  排除次新股: {len(excluded_new)} 只")
    print(f"  获取信息失败: {len(excluded_error)} 只")
    
    if excluded_st:
        print(f"\n排除的ST股:")
        for code, name in excluded_st:
            print(f"    {code}: {name}")
    
    if excluded_new:
        print(f"\n排除的次新股:")
        for code, name, days in excluded_new:
            print(f"    {code} ({name}): 上市{days}天")
    
    return filtered_stocks

def get_billboard_stocks(start_date='20240101', end_date='20240930', min_frequency=1):
    """
    获取龙虎榜股票代码
    
    参数:
    start_date: 开始日期
    end_date: 结束日期  
    min_frequency: 最小上榜次数，筛选活跃股票
    
    返回:
    活跃股票代码列表
    """
    print(f"正在获取龙虎榜数据...")
    print(f"时间范围: {start_date} - {end_date}")
    print(f"最小上榜次数: {min_frequency}")
    
    try:
        # 获取龙虎榜数据
        billboard_data = ef.stock.get_daily_billboard(
            start_date=start_date, 
            end_date=end_date
        )
        
        if billboard_data is None or billboard_data.empty:
            print("警告: 未获取到龙虎榜数据，使用默认股票列表")
            return [
                '600519',  # 贵州茅台
                '000001',  # 平安银行
                '000002',  # 万科A
                '600036',  # 招商银行
                '000858',  # 五粮液
                '002415',  # 海康威视
            ]
        
        print(f"获取到 {len(billboard_data)} 条龙虎榜记录")
        
        # 统计股票上榜频次
        if '股票代码' in billboard_data.columns:
            stock_code_col = '股票代码'
        elif 'code' in billboard_data.columns:
            stock_code_col = 'code'
        elif '代码' in billboard_data.columns:
            stock_code_col = '代码'
        else:
            # 如果找不到代码列，打印列名帮助调试
            print(f"龙虎榜数据列名: {billboard_data.columns.tolist()}")
            print("未找到股票代码列，使用默认股票列表")
            return [
                '600519', '000001', '000002', '600036', '000858', '002415'
            ]
        
        # 统计每只股票的上榜次数
        stock_frequency = billboard_data[stock_code_col].value_counts()
        
        # 筛选上榜次数大于等于min_frequency的股票
        active_stocks = stock_frequency[stock_frequency >= min_frequency].index.tolist()
        
        # 清理股票代码格式（移除可能的后缀）
        cleaned_stocks = []
        for stock in active_stocks:
            # 移除可能的.SH或.SZ后缀
            clean_code = str(stock).replace('.SH', '').replace('.SZ', '')
            if len(clean_code) == 6 and clean_code.isdigit():
                cleaned_stocks.append(clean_code)
        
        # 限制股票数量，避免分析时间过长
        max_stocks = 30
        selected_stocks = cleaned_stocks[:max_stocks]
        
        print(f"龙虎榜活跃股票统计:")
        print(f"  总上榜股票数: {len(stock_frequency)}")
        print(f"  活跃股票数 (>={min_frequency}次): {len(active_stocks)}")
        print(f"  最终选择股票数: {len(selected_stocks)}")
        
        print(f"选中的龙虎榜活跃股票:")
        for i, stock in enumerate(selected_stocks, 1):
            frequency = stock_frequency.get(stock, 0)
            print(f"  {i:2d}. {stock} (上榜{frequency}次)")
        
        if not selected_stocks:
            print("警告: 未找到符合条件的龙虎榜股票，使用默认股票列表")
            selected_stocks = [
                '600519', '000001', '000002', '600036', '000858', '002415'
            ]
        
        # 过滤ST股和次新股
        filtered_stocks = filter_stocks(
            selected_stocks, 
            exclude_st=True,           # 排除ST股
            exclude_new_stock=True,    # 排除次新股
            new_stock_days=365         # 上市不到1年的算作次新股
        )
        
        if not filtered_stocks:
            print("警告: 过滤后没有股票，使用部分原始股票")
            return selected_stocks[:5]
        
        return filtered_stocks
        
    except Exception as e:
        print(f"获取龙虎榜数据失败: {e}")
        print("使用默认股票列表")
        return [
            '600519',  # 贵州茅台
            '000001',  # 平安银行  
            '000002',  # 万科A
            '600036',  # 招商银行
            '000858',  # 五粮液
            '002415',  # 海康威视
        ]

def add_price_features(df):
    """
    添加额外的价格相关特征
    """
    # 价格相对指标
    df['Close_Open_Ratio'] = df['Close'] / df['Open']
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_High_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    
    # 价格变化率特征
    for period in [1, 3, 5, 10, 20]:
        df[f'Close_Pct_{period}'] = df['Close'].pct_change(periods=period)
        df[f'Volume_Pct_{period}'] = df['Volume'].pct_change(periods=period)
        df[f'TurnoverRate_Pct_{period}'] = df['TurnoverRate'].pct_change(periods=period)
    
    # 价格波动相关特征
    df['Price_Volatility_5'] = df['Close'].pct_change().rolling(window=5).std()
    df['Price_Volatility_10'] = df['Close'].pct_change().rolling(window=10).std()
    df['Price_Volatility_20'] = df['Close'].pct_change().rolling(window=20).std()
    
    # 成交量价格关系
    df['Volume_Price_Correlation'] = df['Volume'].rolling(window=20).corr(df['Close'])
    df['Turnover_Price_Correlation'] = df['TurnoverRate'].rolling(window=20).corr(df['Close'])
    
    # 资金流相关比率
    if 'MainNetInflow' in df.columns:
        df['MainInflow_Volume_Ratio'] = df['MainNetInflow'] / (df['Volume'] + 1)
        df['MainInflow_Amount_Ratio'] = df['MainNetInflow'] / (df['Amount'] + 1)
        
        # 资金流动量
        for period in [3, 5, 10]:
            df[f'MainInflow_MA_{period}'] = df['MainNetInflow'].rolling(window=period).mean()
            df[f'MainInflowRatio_MA_{period}'] = df['MainNetInflowRatio'].rolling(window=period).mean()
    
    return df

def download_china_stock_enhanced_data(stock_codes, start_date='20240101', end_date='20250930', 
                                       save_to_file=True, min_data_points=180):
    """
    下载增强版中国股票数据，包含技术指标和资金流数据
    
    参数:
    stock_codes: 股票代码列表
    start_date: 开始日期
    end_date: 结束日期
    save_to_file: 是否保存到文件
    min_data_points: 最少数据点数量，低于此数量的股票将被排除
    
    返回:
    all_data: 字典，键为股票代码，值为DataFrame
    """
    print(f"正在下载 {len(stock_codes)} 只中国股票的增强数据...")
    print(f"股票代码: {stock_codes}")
    print(f"时间范围: {start_date} - {end_date}")
    print(f"最少数据点要求: {min_data_points} 条")
    
    all_data = {}
    insufficient_data_stocks = []
    
    for stock_code in stock_codes:
        print(f"\n处理股票: {stock_code}")
        
        try:
            # 1. 获取基础K线数据
            print(f"  获取 {stock_code} 的K线数据...")
            kline_data = ef.stock.get_quote_history(
                stock_codes=[stock_code], 
                beg=start_date, 
                end=end_date
            )
            
            if not isinstance(kline_data, dict) or stock_code not in kline_data:
                print(f"  警告: {stock_code} K线数据获取失败")
                continue
                
            df = kline_data[stock_code]
            if df.empty:
                print(f"  警告: {stock_code} K线数据为空")
                continue
            
            # 重命名列以符合标准格式
            df = df.rename(columns={
                '日期': 'Date',
                '开盘': 'Open',
                '收盘': 'Close',
                '最高': 'High',
                '最低': 'Low',
                '成交量': 'Volume',
                '成交额': 'Amount',
                '涨跌幅': 'PriceChangeRate',
                '涨跌额': 'PriceChangeAmount',
                '换手率': 'TurnoverRate',
                '振幅': 'Amplitude'
            })
            
            # 设置日期为索引
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            print(f"  获取到 {len(df)} 条K线记录")
            
            # 2. 获取资金流数据
            print(f"  获取 {stock_code} 的资金流数据...")
            try:
                money_flow = ef.stock.get_history_bill(stock_code)
                if isinstance(money_flow, pd.DataFrame) and not money_flow.empty:
                    # 处理资金流数据
                    money_flow['日期'] = pd.to_datetime(money_flow['日期'])
                    money_flow.set_index('日期', inplace=True)
                    
                    # 重命名资金流列
                    money_flow_columns = {
                        '主力净流入': 'MainNetInflow',
                        '主力净流入占比': 'MainNetInflowRatio',
                        '超大单净流入': 'SuperLargeNetInflow',
                        '超大单流入净占比': 'SuperLargeNetInflowRatio',
                        '大单净流入': 'LargeNetInflow',
                        '大单流入净占比': 'LargeNetInflowRatio',
                        '中单净流入': 'MediumNetInflow',
                        '中单流入净占比': 'MediumNetInflowRatio',
                        '小单净流入': 'SmallNetInflow',
                        '小单流入净占比': 'SmallNetInflowRatio'
                    }
                    
                    for old_col, new_col in money_flow_columns.items():
                        if old_col in money_flow.columns:
                            money_flow[new_col] = money_flow[old_col]
                    
                    # 合并资金流数据到主数据框
                    df = df.join(money_flow[list(money_flow_columns.values())], how='left')
                    print(f"  成功合并 {len(money_flow)} 条资金流记录")
                else:
                    print(f"  警告: {stock_code} 资金流数据为空")
                    # 填充缺失的资金流列
                    money_flow_cols = ['MainNetInflow', 'MainNetInflowRatio', 'SuperLargeNetInflow', 
                                     'SuperLargeNetInflowRatio', 'LargeNetInflow', 'LargeNetInflowRatio',
                                     'MediumNetInflow', 'MediumNetInflowRatio', 'SmallNetInflow', 'SmallNetInflowRatio']
                    for col in money_flow_cols:
                        df[col] = 0
                        
            except Exception as e:
                print(f"  资金流数据获取异常: {e}")
                # 填充缺失的资金流列
                money_flow_cols = ['MainNetInflow', 'MainNetInflowRatio', 'SuperLargeNetInflow', 
                                 'SuperLargeNetInflowRatio', 'LargeNetInflow', 'LargeNetInflowRatio',
                                 'MediumNetInflow', 'MediumNetInflowRatio', 'SmallNetInflow', 'SmallNetInflowRatio']
                for col in money_flow_cols:
                    df[col] = 0
            
            # 3. 添加技术指标
            print(f"  计算 {stock_code} 的技术指标...")
            try:
                df = add_technical_indicators(df)
                print(f"  技术指标计算完成")
            except Exception as e:
                print(f"  技术指标计算异常: {e}")
                # 继续处理，即使技术指标计算失败
            
            # 4. 添加额外的价格特征
            df = add_price_features(df)
            
            # 5. 数据清理
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(method='ffill').fillna(0)
            
            # 6. 检查数据记录数量
            if len(df) < min_data_points:
                print(f"  ⚠ 警告: {stock_code} 数据点不足 ({len(df)} < {min_data_points})，已排除")
                insufficient_data_stocks.append((stock_code, len(df)))
                continue
            
            all_data[stock_code] = df
            print(f"  ✅ {stock_code} 数据处理完成，数据记录: {len(df)} 条，特征数: {df.shape[1]}")
            
            # 可选：保存单个股票数据
            if save_to_file:
                filename = f'data/stock_{stock_code}_data.csv'
                df.to_csv(filename)
                print(f"  已保存到: {filename}")
            
        except Exception as e:
            print(f"  错误: {stock_code} 数据处理失败 - {e}")
            continue
    
    print(f"\n总共成功处理 {len(all_data)} 只股票的数据")
    
    if insufficient_data_stocks:
        print(f"\n因数据记录不足被排除的股票 ({len(insufficient_data_stocks)} 只):")
        for code, count in insufficient_data_stocks:
            print(f"  {code}: 仅有 {count} 条记录 (要求≥{min_data_points})")
    
    # 保存汇总数据
    if save_to_file and all_data:
        print("\n保存汇总数据...")
        import pickle
        with open('data/all_stock_data.pkl', 'wb') as f:
            pickle.dump(all_data, f)
        print("  已保存到: data/all_stock_data.pkl")
    
    return all_data

def main():
    """
    数据下载主函数
    """
    from datetime import datetime
    import os
    
    print("="*80)
    print("股票数据下载工具")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建数据目录
    if not os.path.exists('data'):
        os.makedirs('data')
        print("创建数据目录: data/")
    
    # 1. 获取龙虎榜活跃股票
    print("\n步骤1: 获取龙虎榜活跃股票")
    print("-" * 50)
    billboard_start_date = '2025-01-01'
    billboard_end_date = '2025-09-30'
    
    china_stocks = get_billboard_stocks(
        start_date=billboard_start_date,
        end_date=billboard_end_date,
        min_frequency=2
    )
    
    print(f"\n最终选择分析股票: {china_stocks}")
    print(f"股票数量: {len(china_stocks)}")
    
    # 2. 下载股票数据
    print(f"\n步骤2: 下载股票增强数据")
    print("-" * 50)
    analysis_start_date = '20240101'
    analysis_end_date = '20250930'
    
    stock_data = download_china_stock_enhanced_data(
        china_stocks, 
        start_date=analysis_start_date, 
        end_date=analysis_end_date,
        save_to_file=True,
        min_data_points=180  # 至少需要180个交易日（约9个月）的数据
    )
    
    if not stock_data:
        print("\n错误: 没有获取到任何股票数据")
        return
    
    # 3. 保存股票列表
    print(f"\n步骤3: 保存股票列表")
    print("-" * 50)
    billboard_stocks_df = pd.DataFrame({
        'stock_code': china_stocks,
        'download_date': datetime.now().strftime('%Y-%m-%d'),
        'data_start_date': analysis_start_date,
        'data_end_date': analysis_end_date,
        'source': 'billboard_active_stocks'
    })
    billboard_stocks_df.to_csv('data/stock_list.csv', index=False)
    print("已保存股票列表到: data/stock_list.csv")
    
    print("\n" + "="*80)
    print("数据下载完成！")
    print("="*80)
    print(f"成功下载 {len(stock_data)} 只股票的数据")
    print(f"数据保存位置: data/")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n下一步: 运行 stock_tsfresh_analysis.py 进行数据分析")

if __name__ == '__main__':
    main()

