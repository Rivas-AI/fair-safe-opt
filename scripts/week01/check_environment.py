from __future__ import annotations

import importlib
import platform


REQUIRED = ["numpy", "pandas", "sklearn", "torch", "yaml"]


def main() -> None:
    print(f"python: {platform.python_version()}")
    for name in REQUIRED:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print(f"{name}: {version}")


if __name__ == "__main__":
    main()
