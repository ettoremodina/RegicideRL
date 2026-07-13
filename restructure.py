import os
import shutil
import glob

def safe_move(src_pattern, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for src in glob.glob(src_pattern):
        if os.path.exists(src):
            print(f"Moving {src} to {dst_dir}")
            try:
                shutil.move(src, dst_dir)
            except Exception as e:
                print(f"Error moving {src}: {e}")

def main():
    base_dir = r"c:\Users\modin\Desktop\programming\GAMES\Regicide"
    os.chdir(base_dir)

    # 1. Old tests
    old_test_files = [
        "tests/detailed_game_inspector.py",
        "tests/policy_analysis.py",
        "tests/quick_policy_analysis.py",
        "tests/random_policy_evaluation.py",
        "tests/simple_action_test.py",
        "tests/test_enhanced_policy.py",
        "tests/test_enhanced_policy_fix.py",
        "tests/test_rule_based_policy.py",
        "tests/basic_test.py",
        "tests/test_env*.py"
    ]
    for pattern in old_test_files:
        safe_move(pattern, "archive/old_tests")

    # 2. Config & test flow
    safe_move("game/test_new_flow.py", "archive")
    safe_move("config.py", "archive")
    safe_move("legacy", "archive")

    # 3. Directories
    safe_move("policy", "archive/policy")
    safe_move("train", "archive/train")
    safe_move("reports", "archive/reports")

if __name__ == "__main__":
    main()
