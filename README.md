# rqre-ovi

Code implementation for the paper "Strategically Robust Multi-Agent Reinforcement Learning with Linear Function Approximation".

## Install

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Run Overcooked

```bash
python -u rqre-ovi/overcooked/train.py --algorithm qre
python -u rqre-ovi/overcooked/train.py --algorithm rqe
python -u rqre-ovi/overcooked/train.py --algorithm nqovi
```

## Run StagHunt

```bash
python -u rqre-ovi/staghunt/train.py --algorithm qre
python -u rqre-ovi/staghunt/train.py --algorithm rqe
python -u rqre-ovi/staghunt/train.py --algorithm nqovi
```

## Note

The `overcooked_jaxmarl` directory is external environment code adapted from the JaxMARL project and is not original to this repository:

https://github.com/FLAIROx/JaxMARL