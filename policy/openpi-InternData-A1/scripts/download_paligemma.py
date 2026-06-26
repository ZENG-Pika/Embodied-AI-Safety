import os
from pathlib import Path

def download_from_gcs(gcs_uri: str, local_path: str):
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if os.system("which gsutil > /dev/null 2>&1") == 0:
        cmd = f"gsutil cp {gcs_uri} {local_path}"
    else:
        gcs_http = gcs_uri.replace("gs://", "https://storage.googleapis.com/")
        cmd = f"wget -O {local_path} {gcs_http}"

    print(f"⬇️  Executing: {cmd}")
    ret = os.system(cmd)
    if ret == 0:
        print("✅ Download complete:", local_path)
    else:
        raise RuntimeError(f"Download failed: {gcs_uri}")

    return local_path


if __name__ == "__main__":
    gcs_uri = "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz"
    save_path = "checkpoints/jax/paligemma/pt_224.npz"
    download_from_gcs(gcs_uri, save_path)