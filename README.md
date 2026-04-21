# Local_RLCT_Calculater

## ディレクトリ構成

これから作るものの設計図

```text
Local_RLCT_Calculater/
├── README.md                  # プロジェクト概要と使い方
├── .gitignore         
├── gpu-test.ipynb             # GPU使用可能確認
├── local_rlct_estimater.py    # local_rclt推定用のコード  
├── quadratic_function.py      # 動作確認用の二次関数         
└── main.ipynb                 # main実行用 
```



### メモ

必要に応じて、役割ごとにもう少し階層を分けて書くこともできます。

```text
Local_RLCT_Calculater/
├── src/                       # アプリケーション本体
│   └── local_rlct_pipeline.py
├── tests/                     # テストコード
│   └── test_sgld_torch_local.py
├── docs/                      # 設計資料・補足ドキュメント
│   └── local_rlct_design.md
├── scripts/                   # 実行補助スクリプト
│   └── training_tasks.py
└── README.md                  # プロジェクト概要
```

ポイント:

- ディレクトリ名の右に `#` で役割を書く
- 深すぎる階層は省略して、重要な部分だけ載せる
- 実際の配置と README の説明をできるだけ一致させる
