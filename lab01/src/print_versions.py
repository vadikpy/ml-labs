"""Print the Python version and every installed package version."""
import platform
import sys
from importlib.metadata import distributions

print(f"Python {sys.version}")
print(f"Platform {platform.platform()}")
for name, version in sorted({(d.metadata["Name"], d.version) for d in distributions()}, key=lambda p: p[0].lower()):
    print(f"{name}=={version}")
