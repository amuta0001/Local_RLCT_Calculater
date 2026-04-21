# Local_RLCT_Calculater

PyTorch ベースで局所 RLCT を試行的に推定するための実験用リポジトリです。  
現在は `common/` 配下に共通の推定器を置き、ノートブックから読み込んで検証する構成になっています。

## ディレクトリ構成

```text
Local_RLCT_Calculater/
├── README.md                           # プロジェクト概要
├──data/
│   └── mnist/                          # MNIST データ
├── mnist.ipynb                         # MNIST 関連の実験ノートブック
├── common/                             # 共通リソース
│   ├── __init__.py
│   ├── local_rlct_estimater.py         # 共通の局所 RLCT 推定器
│   ├── gpu-test.ipynb                  # GPU 利用確認用ノートブック
│   └── cuda_installer.pyz              # CUDA 関連の補助ファイル
├── objective_function/                 # 実験用の目的関数モジュール
│   └── quadratic_function.py           # 二次関数の Torch モデル
└── ex1_quadratic_function.ipynb        # 二次関数での RLCT 推定実験
```

## 使い方

- 共通推定器は `common.local_rlct_estimater` から import します。
- 二次関数の検証は `ex1_quadratic_function.ipynb` で行います。
- 実験ノートブックでは、必要に応じて `importlib.reload(...)` で共通モジュールを再読み込みしてください。

例:

```python
from common import local_rlct_estimater
from objective_function import quadratic_function

LocalRLCTTorchEstimator = local_rlct_estimater.LocalRLCTTorchEstimator
QuadraticTorchModel = quadratic_function.QuadraticTorchModel
```
