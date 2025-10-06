"""
GitHub推送自动化脚本（适配当前股票预测项目）
自动完成Git初始化、添加、提交、推送
包含数据和模型文件（约148MB）
"""

import subprocess
import sys
import os

def print_step(step_num, message):
    """打印步骤信息"""
    print("\n" + "="*80)
    print(f"步骤 {step_num}: {message}")
    print("="*80)

def run_command(command, error_message, check=True):
    """运行命令"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        if result.stdout:
            print(result.stdout)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        if check:
            print(f"❌ {error_message}")
            if e.stderr:
                print(f"错误信息: {e.stderr}")
        return False, e.stderr

def check_git():
    """检查Git是否安装"""
    print_step(1, "检查Git安装")
    
    success, output = run_command("git --version", "Git未安装", check=False)
    
    if success:
        print(f"✅ Git已安装: {output.strip()}")
        return True
    else:
        print("❌ Git未安装")
        print("\n请访问 https://git-scm.com/download/win 下载安装")
        return False

def check_git_config():
    """检查Git配置"""
    print_step(2, "检查Git配置")
    
    success, name = run_command("git config --global user.name", "", check=False)
    success2, email = run_command("git config --global user.email", "", check=False)
    
    if name.strip() and email.strip():
        print(f"✅ 用户名: {name.strip()}")
        print(f"✅ 邮箱: {email.strip()}")
        return True
    else:
        print("⚠️  Git未配置用户信息")
        print("\n请输入配置信息：")
        
        username = input("GitHub用户名: ").strip()
        email_input = input("GitHub邮箱: ").strip()
        
        if username and email_input:
            run_command(f'git config --global user.name "{username}"', "配置失败")
            run_command(f'git config --global user.email "{email_input}"', "配置失败")
            print("✅ 配置成功")
            return True
        else:
            print("❌ 配置信息不能为空")
            return False

def check_required_files():
    """检查必要文件"""
    print_step(3, "检查项目文件")
    
    # 当前项目的核心文件
    required_files = [
        'app_streamlit.py',  # 主应用
        'train_ma20_multi_horizon.py',  # MA20训练
        'multi_horizon_prediction_system.py',  # 形态识别训练
    ]
    
    config_files = [
        'requirements_streamlit.txt',
        '.gitignore',
        'README.md',
    ]
    
    all_ok = True
    
    print("\n核心应用文件:")
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - 文件不存在")
            all_ok = False
    
    print("\n配置文件:")
    for file in config_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ⚠️  {file} - 建议创建")
    
    # 检查数据目录
    if os.path.exists('data'):
        data_files = [f for f in os.listdir('data') if f.endswith(('.pkl', '.csv'))]
        if data_files:
            total_size = sum(os.path.getsize(os.path.join('data', f)) for f in data_files) / (1024*1024)
            print(f"\n数据文件: ✅ 找到 {len(data_files)} 个文件 (约{total_size:.1f}MB)")
        else:
            print("\n数据文件: ⚠️  data/文件夹存在但没有数据文件")
    else:
        print("\n数据文件: ⚠️  data/文件夹不存在")
    
    # 检查模型目录
    if os.path.exists('models'):
        pkl_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
        if pkl_files:
            total_size = sum(os.path.getsize(os.path.join('models', f)) for f in pkl_files) / (1024*1024)
            print(f"模型文件: ✅ 找到 {len(pkl_files)} 个模型 (约{total_size:.1f}MB)")
            
            # 检查是否有超过100MB的文件
            for pkl in pkl_files:
                size = os.path.getsize(os.path.join('models', pkl)) / (1024*1024)
                if size > 100:
                    print(f"  ⚠️  {pkl}: {size:.2f}MB (超过100MB，GitHub可能拒绝)")
        else:
            print("模型文件: ⚠️  models/文件夹存在但没有模型文件")
    else:
        print("模型文件: ⚠️  models/文件夹不存在")
    
    return True  # 不强制要求所有文件

def clean_temp_files():
    """清理临时文件"""
    print_step(4, "清理临时文件")
    
    temp_patterns = [
        'test_chinese_display.py',
        'test_pattern_recognition.py',
        'test_streamlit.py',
        'test_pattern_result.csv',
        'clean_emojis.py',
        'fix_matplotlib_chinese.py',
        'check_system_status.py',
    ]
    
    cleaned = 0
    for pattern in temp_patterns:
        if os.path.exists(pattern):
            try:
                os.remove(pattern)
                print(f"  ✅ 已删除: {pattern}")
                cleaned += 1
            except:
                print(f"  ⚠️  无法删除: {pattern}")
    
    # 清理__pycache__
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            try:
                import shutil
                shutil.rmtree(pycache_path)
                print(f"  ✅ 已删除: {pycache_path}")
                cleaned += 1
            except:
                pass
    
    if cleaned > 0:
        print(f"\n✅ 清理完成，共删除 {cleaned} 项")
    else:
        print("\n✅ 没有需要清理的文件")
    
    return True

def git_init():
    """初始化Git仓库"""
    print_step(5, "初始化Git仓库")
    
    if os.path.exists('.git'):
        print("⚠️  Git仓库已存在")
        choice = input("是否重新初始化？(y/n): ").strip().lower()
        if choice == 'y':
            print("\n正在删除旧的Git仓库...")
            if sys.platform == 'win32':
                result = subprocess.run('rmdir /s /q .git', shell=True, capture_output=True)
                if result.returncode != 0:
                    print("⚠️  无法自动删除.git文件夹，请手动删除后重试")
                    return False
            else:
                import shutil
                try:
                    shutil.rmtree('.git')
                except Exception as e:
                    print(f"❌ 删除失败: {e}")
                    return False
            print("✅ 旧仓库已删除")
        else:
            print("使用现有Git仓库")
            success, _ = run_command("git status", "", check=False)
            if not success:
                print("\n❌ 现有Git仓库已损坏！建议重新初始化")
                return False
            return True
    
    success, _ = run_command("git init", "初始化失败")
    
    if success:
        print("✅ Git仓库初始化成功")
        run_command("git branch -M main", "创建分支失败", check=False)
        return True
    return False

def git_add():
    """添加文件到暂存区"""
    print_step(6, "添加文件到Git")
    
    print("\n⚠️  即将添加所有文件（包括数据和模型，约148MB）")
    print("这可能需要30-60秒...")
    
    choice = input("\n继续？(y/n): ").strip().lower()
    if choice != 'y':
        print("取消添加")
        return False
    
    print("\n添加所有文件...")
    success, _ = run_command("git add .", "添加文件失败")
    
    if success:
        print("\n查看将要提交的文件状态:")
        run_command("git status -s", "查看状态失败", check=False)
        
        print("\n✅ 文件已添加")
        return True
    
    return False

def git_commit():
    """提交到本地仓库"""
    print_step(7, "提交到本地仓库")
    
    print("\n请输入提交说明（留空使用默认）:")
    commit_msg = input("提交说明: ").strip()
    
    if not commit_msg:
        commit_msg = "Initial commit: Stock prediction system with data and models"
    
    success, _ = run_command(f'git commit -m "{commit_msg}"', "提交失败")
    
    if success:
        print("✅ 提交成功")
        return True
    else:
        print("⚠️  提交失败")
        return False

def git_remote():
    """配置远程仓库"""
    print_step(8, "配置GitHub远程仓库")
    
    success, output = run_command("git remote -v", "", check=False)
    
    if success and output.strip():
        print("\n当前远程仓库:")
        print(output)
        choice = input("\n是否重新配置？(y/n): ").strip().lower()
        if choice != 'y':
            return True
        else:
            run_command("git remote remove origin", "", check=False)
    
    print("\n" + "="*80)
    print("请先在GitHub创建仓库！")
    print("="*80)
    print("\n步骤:")
    print("  1. 访问 https://github.com/new")
    print("  2. 填写仓库名称（如：stock-prediction-system）")
    print("  3. 选择Public或Private")
    print("  4. 不要勾选任何初始化选项")
    print("  5. 点击 Create repository")
    print("  6. 复制仓库地址")
    print()
    
    repo_url = input("GitHub仓库URL: ").strip()
    
    if not repo_url:
        print("❌ 仓库URL不能为空")
        return False
    
    success, _ = run_command(f"git remote add origin {repo_url}", "添加远程仓库失败")
    
    if success:
        print("✅ 远程仓库配置成功")
        return True
    return False

def git_push():
    """推送到GitHub"""
    print_step(9, "推送到GitHub")
    
    print("\n" + "="*80)
    print("重要提示")
    print("="*80)
    print("\n即将上传约148MB数据到GitHub")
    print("预计需要5-10分钟（取决于网速）")
    print()
    print("上传内容:")
    print("  - 代码文件 (~2MB)")
    print("  - 数据文件 (~36MB)")
    print("  - 模型文件 (~110MB)")
    print()
    print("登录信息:")
    print("  - 用户名: 你的GitHub用户名")
    print("  - 密码: Personal Access Token（不是密码！）")
    print()
    print("如何获取Token:")
    print("  1. 访问 https://github.com/settings/tokens")
    print("  2. Generate new token (classic)")
    print("  3. 勾选 'repo'")
    print("  4. 复制Token")
    print()
    
    choice = input("准备好推送了吗？(y/n): ").strip().lower()
    
    if choice != 'y':
        print("取消推送")
        return False
    
    print("\n开始推送（请保持网络连接）...")
    print("="*80)
    success, output = run_command("git push -u origin main", "推送失败", check=False)
    
    if success or "Everything up-to-date" in output:
        print("\n" + "="*80)
        print("✅ 推送成功！")
        print("="*80)
        print("\n请在浏览器访问你的GitHub仓库查看")
        return True
    else:
        print("\n❌ 推送失败")
        print("\n常见问题:")
        print("  1. 凭据错误: 使用Personal Access Token而不是密码")
        print("  2. 权限不足: 检查仓库URL是否正确")
        print("  3. 网络问题: 检查网络连接")
        print("  4. 文件太大: 检查是否有单个文件超过100MB")
        return False

def main():
    """主函数"""
    print()
    print("="*80)
    print("  📦 GitHub自动上传工具（适配当前项目）")
    print("="*80)
    print()
    print("此工具将帮助你:")
    print("  1. 清理临时文件")
    print("  2. 初始化Git仓库")
    print("  3. 添加所有文件（包括数据和模型）")
    print("  4. 提交并推送到GitHub")
    print()
    print("注意事项:")
    print("  - 将上传约148MB数据")
    print("  - 需要5-10分钟")
    print("  - 需要GitHub Personal Access Token")
    print()
    
    choice = input("是否继续？(y/n): ").strip().lower()
    if choice != 'y':
        print("取消操作")
        return
    
    # 执行步骤
    steps = [
        (check_git, "检查Git"),
        (check_git_config, "配置Git"),
        (check_required_files, "检查文件"),
        (clean_temp_files, "清理临时文件"),
        (git_init, "初始化Git"),
        (git_add, "添加文件"),
        (git_commit, "提交"),
        (git_remote, "配置远程"),
        (git_push, "推送"),
    ]
    
    for step_func, step_name in steps:
        if not step_func():
            print(f"\n❌ {step_name}失败，推送中止")
            print("\n请解决问题后重新运行此脚本")
            return
    
    print("\n" + "="*80)
    print("  ✅ 上传完成！")
    print("="*80)
    print()
    print("🎉 代码已成功推送到GitHub！")
    print()
    print("📋 下一步:")
    print("  1. 访问你的GitHub仓库检查文件")
    print("  2. 测试克隆:")
    print("     git clone [你的仓库URL]")
    print("     cd [仓库目录]")
    print("     pip install -r requirements_streamlit.txt")
    print("     streamlit run app_streamlit.py")
    print()
    print("  3. 查看详细文档:")
    print("     - README.md")
    print("     - COMPLETE_RETRAINING_GUIDE.md")
    print("     - TWO_MODEL_SYSTEMS_COMPARISON.md")
    print()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  操作被用户中断")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

