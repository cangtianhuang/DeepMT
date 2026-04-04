"""pytest 配置：确保子目录测试可被发现，并将项目根加入 sys.path"""

import sys
from pathlib import Path

# 将项目根目录加入 sys.path，无需手动设置 PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))
