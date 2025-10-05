# ML Ops Assignment 1

This repo implements the assignment with:
- `train.py` — **DecisionTreeRegressor** (uses shared utils in `misc.py`)
- `train2.py` — **KernelRidge** (reuses the same `misc.py`)
- `.github/workflows/ci.yml` — GitHub Actions workflow that runs on pushes to **kernelridge** and prints both models' Test MSE.

---

## Setup

```bash
# create and activate environment
conda create -n mlops-a1 python=3.11 -y
conda activate mlops-a1

# install deps
pip install -r requirements.txt

python train.py

python train2.py

[DecisionTreeRegressor] Test MSE: 10.4161
[KernelRidge] Test MSE: 18.1512

Branches

main — merged code + README

dtree — DecisionTreeRegressor work (requirements.txt, misc.py, train.py)

kernelridge — KernelRidge model + CI workflow (train2.py, .github/workflows/ci.yml)

Do not delete any branch (per assignment).
CI/CD (GitHub Actions)

Workflow: .github/workflows/ci.yml

Triggers on push to kernelridge:

Set up Python 3.11

Install dependencies

Run both train.py and train2.py

Print Test MSE in logs (visible in Actions tab)
| Model                 | Test MSE |
| --------------------- | -------- |
| DecisionTreeRegressor | 10.4161  |
| KernelRidge           | 18.1512  |

Submission Checklist

Include in your report:

Repo link: https://github.com/Saurabh1220/mlops-assignment-1

Table with both Test MSE values (above)

Screenshot of Branches (main, dtree, kernelridge)

Screenshot of Actions log showing both models’ MSE outputs
