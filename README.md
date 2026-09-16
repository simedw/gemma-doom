# Gemma Doom

Gemma 4 E2B controls Doom from pixels, locally on a single GPU. No fine-tuning. It sort of works.

Requires Linux, [uv](https://docs.astral.sh/uv/), and an NVIDIA GPU. Tested on an RTX 4090.

## Run

```bash
git clone https://github.com/simedw/gemma-doom.git
cd gemma-doom
uv sync --locked
uv run typesafe-gemma download
uv run typesafe-doom --gpu 0
```

Open [localhost:8766](http://127.0.0.1:8766). Wait for the model to finish loading, select **Gemma**, and press **Play**.

Choose **You** to play yourself or **Copilot** to see the model's suggestions while you play.
