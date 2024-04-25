echo "Installing Python packages."

pip install -r requirements.txt

echo "Package installation complete."

echo "Checking whether transcoders present."
DIR_NAME='./gpt-2-small-transcoders'
if [ ! -d "$DIR_NAME" ]; then
    mkdir "$DIR_NAME"
fi
if [ -z $(ls -A "$DIR_NAME") ]; then
    echo "Transcoders not found. Downloading transcoders."

    export HF_HUB_DISABLE_PROGRESS_BARS=1
    python - <<HERE
from huggingface_hub import snapshot_download
snapshot_download(repo_id="pchlenski/gpt2-transcoders", allow_patterns=["*.pt"],
    local_dir="$DIR_NAME", local_dir_use_symlinks=False
)
HERE
    export HF_HUB_DISABLE_PROGRESS_BARS=0
    echo "Transcoders downloaded."
fi

