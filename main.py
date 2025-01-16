# This is mainly for testing changes without having to package and install.

import time

from src.frequensee import main_cli


if __name__ == "__main__":
    start = time.perf_counter()
    main_cli()
    print("Time: ",time.perf_counter() - start)