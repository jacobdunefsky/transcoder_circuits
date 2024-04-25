echo "Installing Python packages."

pip -r requirements.txt

echo "Package installation complete."

echo "Checking whether transcoders present."
DIR_NAME='./new-gpt-2-small-transcoders'
if [ ! -d "$DIR_NAME" ]; then
    mkdir "$DIR_NAME"
fi
DIR_NAME='./new-gpt-2-small-transcoders'
if [ -z $(ls -A "$DIR_NAME") ]; then
    echo "Transcoders not found. Downloading transcoders."

    python - <<HERE
from huggingface_hub import snapshot_download
snapshot_download(repo_id="pchlenski/gpt2-transcoders", allow_patterns=["*.pt"],
    local_dir="$DIR_NAME", local_dir_use_symlinks=False
)
HERE
    echo "Transcoders downloaded."
fi

