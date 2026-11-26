"""downloads the text encoder's adapters and the kimodo checkpoint into models/hf."""

import glob
import os

from huggingface_hub import hf_hub_download, snapshot_download

import install


def main():
    hub = os.path.join(os.environ["HF_HOME"], "hub")
    for repo in (*install.ADAPTERS, install.KIMODO_MODEL):
        # hf_hub starts a cut-off file over, so drop what an interrupted run left
        for left in glob.glob(os.path.join(hub, "models--" + repo.replace("/", "--"), "blobs", "*.incomplete")):
            os.remove(left)
    print("Kimodo model and the text encoder's adapters (1.5 GB)", flush=True)
    for repo, files in install.ADAPTERS.items():
        for name in files:
            hf_hub_download(repo, name)
    snapshot_download(install.KIMODO_MODEL)


if __name__ == "__main__":
    main()
