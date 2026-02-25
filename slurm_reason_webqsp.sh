#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -J reason_webqsp
#SBATCH -A computerlab-sl2-gpu
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module load miniconda/3
source /usr/local/software/master/miniconda/3/etc/profile.d/conda.sh

# Create env once; harmless if it already exists
conda create -n reasoner python=3.10.14 -y || true
conda activate reasoner

pip install -U huggingface_hub
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.5.5 openai==1.50.2 wandb

# Download preprocessed artifacts (cached if re-run)
huggingface-cli download siqim311/SubgraphRAG --revision main --local-dir ./

cd reason
python main.py -d webqsp --prompt_mode scored_100
