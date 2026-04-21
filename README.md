# Local_RLCT_Calculater

PyTorch ベースで局所 RLCT を試行的に推定するための実験用リポジトリです。  
現在は `common/` 配下に共通の推定器を置き、`objective_function/` に実験用モデルを置き、ノートブックから読み込んで検証する構成になっています。

## ディレクトリ構成

```text
Local_RLCT_Calculater/
├── README.md                           # プロジェクト概要
├── data/
│   └── mnist/                          # MNIST データ
├── mnist.ipynb                         # MNIST 関連の実験ノートブック
├── common/                             # 共通リソース
│   ├── __init__.py
│   ├── local_rlct_estimater.py         # 共通の局所 RLCT 推定器
│   └── gpu-test.ipynb                  # GPU 利用確認用ノートブック
├── objective_function/                 # 実験用の目的関数モジュール
│   ├── quadratic_function.py           # 二次関数の Torch モデル
│   └── gelu_dnn.py                     # GELU ネットワークの真のモデルと学習モデル
├── ex1_quadratic_function.ipynb        # 二次関数での RLCT 推定実験
└── ex2_linear_dnn.ipynb                # 線形 DNN でのデータ生成・学習・RLCT 推定実験
```

## 使い方

- 共通推定器は `common.local_rlct_estimater` から import します。
- 二次関数の検証は `ex1_quadratic_function.ipynb` で行います。
- 線形 DNN の検証は `ex2_linear_dnn.ipynb` で行います。
- 実験ノートブックでは、必要に応じて `importlib.reload(...)` で共通モジュールを再読み込みしてください。

例:

```python
from common import local_rlct_estimater
import objective_function.quadratic_function as quadratic_function
import objective_function.gelu_dnn as gelu_dnn

LocalRLCTTorchEstimator = local_rlct_estimater.LocalRLCTTorchEstimator
QuadraticTorchModel = quadratic_function.QuadraticTorchModel
LinearDNNModel = gelu_dnn.LinearDNNModel
TrueLinearDNN = gelu_dnn.TrueLinearDNN
```
