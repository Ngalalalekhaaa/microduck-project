"""Adjust selected hip-roll HOME; require explicit servo IDs for each operation."""
import sys

from adjust_joint11_home import main as legacy_main


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not any(arg == '--ids' or arg.startswith('--ids=') for arg in argv):
        if '-h' not in argv and '--help' not in argv:
            raise ValueError('必须显式指定 --ids；duck1右髋侧倾使用 --ids 12')
    return legacy_main(argv)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as exc:
        sys.exit(str(exc))
