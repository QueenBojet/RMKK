import sys
from pathlib import Path

def find_project_root():
    """
    查找项目根目录，支持开发环境和PyInstaller打包环境
    
    开发环境：查找包含 config/ 或 src/ 的目录
    打包环境：使用 sys._MEIPASS (PyInstaller设置的临时目录)
    """
    # PyInstaller打包后的环境
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # 打包后运行时，资源文件在 sys._MEIPASS 目录中
        return Path(sys._MEIPASS)
    
    # 开发环境
    current = Path(__file__).resolve()

    # 从当前文件向上搜索项目根目录
    search_paths = [current.parent] + list(current.parents)

    for parent in search_paths:
        # 查找项目标记（config或src目录）
        if (parent / 'config').exists() and (parent / 'src').exists():
            return parent
        
        # 其他项目标记
        if (parent / 'setup.py').exists() or (parent / 'pyproject.toml').exists():
            return parent

        # 到达文件系统根目录
        if parent.parent == parent:
            break

    # 特殊情况处理：如果在 src/utils 目录下，向上两级
    if current.parent.name == 'utils' and current.parent.parent.name == 'src':
        return current.parent.parent.parent

    # 如果在 src 目录下，向上一级
    if current.parent.name == 'src':
        return current.parent.parent

    # 回退：使用当前文件的父目录
    return current.parent


def get_resource_path(relative_path):
    """
    获取资源文件的绝对路径，支持开发和打包环境
    
    Args:
        relative_path: 相对于项目根目录的路径（如 "config/config.yaml"）
    
    Returns:
        Path: 资源文件的绝对路径
    """
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # 打包环境：资源在临时目录
        base_path = Path(sys._MEIPASS)
    else:
        # 开发环境：使用项目根目录
        base_path = PROJECT_ROOT
    
    return base_path / relative_path


# 初始化项目路径（只执行一次）
PROJECT_ROOT = find_project_root()
SRC_DIR = PROJECT_ROOT / 'src' if not getattr(sys, 'frozen', False) else PROJECT_ROOT
CONFIG_DIR = PROJECT_ROOT / 'config'

# 添加到Python路径（仅开发环境需要）
if not getattr(sys, 'frozen', False):
    paths_to_add = [str(PROJECT_ROOT)]
    if SRC_DIR.exists():
        paths_to_add.append(str(SRC_DIR))

    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)


def get_project_path(*args):
    """Get path relative to project root"""
    return PROJECT_ROOT.joinpath(*args)


def get_config_path(*args):
    """Get path relative to config directory"""
    return CONFIG_DIR.joinpath(*args)


def get_src_path(*args):
    """Get path relative to src directory"""
    return SRC_DIR.joinpath(*args)


# Debug function to help diagnose path issues
def debug_paths():
    """Print current path information for debugging"""
    print(f"Current file: {__file__}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Source dir: {SRC_DIR}")
    print(f"Config dir: {CONFIG_DIR}")
    print(f"Python path entries:")
    for i, path in enumerate(sys.path[:5]):  # Show first 5 entries
        print(f"  {i}: {path}")


if __name__ == "__main__":
    debug_paths()