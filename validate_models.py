"""
模型验证脚本
检查新训练的模型是否可用并正常工作
"""
import pickle
import numpy as np
import sys

def validate_ma20_models():
    """验证MA20多时间窗口模型"""
    print("="*80)
    print("验证 MA20 多时间窗口模型")
    print("="*80)
    
    try:
        with open('models/ma20_multi_horizon_models.pkl', 'rb') as f:
            models = pickle.load(f)
        
        print(f"\n[成功] 模型文件加载成功")
        print(f"  包含的时间窗口: {list(models.keys())}")
        
        all_passed = True
        for horizon, model_info in models.items():
            print(f"\n[{horizon}天模型]")
            print(f"  模型类型: {type(model_info['model']).__name__}")
            print(f"  特征数量: {len(model_info['feature_list'])}")
            print(f"  训练样本数: {model_info['train_samples']}")
            print(f"  准确率: {model_info['metrics']['accuracy']:.4f}")
            print(f"  F1分数: {model_info['metrics']['f1_score']:.4f}")
            print(f"  训练时间: {model_info['train_date']}")
            
            # 检查准确率是否在合理范围
            accuracy = model_info['metrics']['accuracy']
            min_acceptable = {1: 0.75, 3: 0.70, 5: 0.65, 10: 0.60}
            if accuracy < min_acceptable.get(horizon, 0.60):
                print(f"  [警告] 准确率低于预期 (最低要求: {min_acceptable.get(horizon, 0.60):.2f})")
                all_passed = False
            
            # 测试预测功能
            model = model_info['model']
            n_features = len(model_info['feature_list'])
            dummy_input = np.random.randn(1, n_features)
            
            try:
                pred = model.predict(dummy_input)
                prob = model.predict_proba(dummy_input)
                print(f"  [成功] 预测功能正常 (预测={pred[0]}, 概率={prob[0, 1]:.4f})")
            except Exception as e:
                print(f"  [错误] 预测功能异常: {e}")
                all_passed = False
        
        print("\n" + "="*80)
        if all_passed:
            print("[成功] MA20模型验证通过")
        else:
            print("[警告] MA20模型验证有问题，请检查")
        print("="*80)
        return all_passed
        
    except FileNotFoundError:
        print(f"\n[错误] 找不到模型文件: models/ma20_multi_horizon_models.pkl")
        print("        请先运行 train_ma20_multi_horizon.py 训练模型")
        return False
    except Exception as e:
        print(f"\n[错误] 验证失败: {e}")
        return False

def validate_pattern_models():
    """验证形态识别模型"""
    print("\n" + "="*80)
    print("验证形态识别模型")
    print("="*80)
    
    try:
        with open('models/multi_horizon_models.pkl', 'rb') as f:
            data = pickle.load(f)
        
        models = data['models']
        print(f"\n[成功] 模型文件加载成功")
        print(f"  包含的时间窗口: {list(models.keys())}")
        
        for horizon in models.keys():
            print(f"\n[{horizon}天模型]")
            print(f"  模型类型: {type(models[horizon]['model']).__name__}")
            print(f"  [成功] 加载正常")
        
        print("\n" + "="*80)
        print("[成功] 形态模型验证通过")
        print("="*80)
        return True
        
    except FileNotFoundError:
        print(f"\n[跳过] 形态识别模型文件不存在")
        print("       如需训练，请运行 multi_horizon_prediction_system.py")
        return True  # 不影响整体验证
    except Exception as e:
        print(f"\n[错误] 验证失败: {e}")
        return False

def validate_data():
    """验证数据文件"""
    print("\n" + "="*80)
    print("验证数据文件")
    print("="*80)
    
    try:
        with open('data/all_stock_data.pkl', 'rb') as f:
            all_data = pickle.load(f)
        
        print(f"\n[成功] 数据文件加载成功")
        print(f"  总股票数: {len(all_data)}")
        
        # 检查前3只股票的数据
        print(f"\n  最新数据日期（前3只）:")
        for code, df in list(all_data.items())[:3]:
            latest_date = df.index[-1].strftime('%Y-%m-%d')
            print(f"    {code}: {latest_date} (共{len(df)}条)")
        
        # 检查必需的列
        first_stock = list(all_data.values())[0]
        required_cols = ['Close', 'MA20', 'Volume', 'RSI', 'MACD']
        missing_cols = [col for col in required_cols if col not in first_stock.columns]
        
        if missing_cols:
            print(f"\n  [警告] 缺少必需的列: {missing_cols}")
            return False
        else:
            print(f"\n  [成功] 所有必需特征都存在")
        
        print("\n" + "="*80)
        print("[成功] 数据验证通过")
        print("="*80)
        return True
        
    except FileNotFoundError:
        print(f"\n[错误] 找不到数据文件: data/all_stock_data.pkl")
        print("        请先运行 stock_data_downloader.py 下载数据")
        return False
    except Exception as e:
        print(f"\n[错误] 验证失败: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "="*80)
    print("           模型验证系统 - 检查所有模型和数据")
    print("="*80)
    print()
    
    # 验证数据
    data_ok = validate_data()
    
    # 验证MA20模型
    ma20_ok = validate_ma20_models()
    
    # 验证形态模型
    pattern_ok = validate_pattern_models()
    
    # 总结
    print("\n" + "="*80)
    print("总体验证结果")
    print("="*80)
    print(f"数据文件: {'✓ 通过' if data_ok else '✗ 失败'}")
    print(f"MA20模型: {'✓ 通过' if ma20_ok else '✗ 失败'}")
    print(f"形态模型: {'✓ 通过' if pattern_ok else '✗ 失败'}")
    print()
    
    if ma20_ok and data_ok:
        print("✓ 核心模型验证通过，可以使用Streamlit应用！")
        print()
        print("运行应用:")
        print("  streamlit run app_streamlit.py")
        sys.exit(0)
    else:
        print("✗ 部分验证失败，请检查上述错误信息！")
        sys.exit(1)

