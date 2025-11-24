uv run --isolated --with "torch>=2.5" --with "uni2ts" \
    --with "torch @ https://download.pytorch.org/whl/cu128/torch-2.7.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl" \
    --with "torchvision @ https://download.pytorch.org/whl/cu128/torchvision-0.22.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl" \
    --with "torchaudio @ https://download.pytorch.org/whl/cu128/torchaudio-2.7.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl" \
    python