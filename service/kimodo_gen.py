"""kimodo_gen with the local llama patches."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import kimodo_patches

if __name__ == "__main__":
    kimodo_patches.apply(template="--no-template-patch" not in sys.argv)
    if "--no-template-patch" in sys.argv:
        sys.argv.remove("--no-template-patch")

    from kimodo.scripts.generate import main

    main()
