# crowd_cpp — 群衆の数え上げと人の位置推定を C++ と Python の両方で

密度マップ回帰（CSRNet）から始めて、人の位置が出せる方式（FIDTM）まで。**学習も推論も評価も両言語**で
でき、成果物は全段 ONNX。姉妹リポの [yolo_lpr_cpp](https://github.com/yomei-o/yolo_lpr_cpp) と同じ設計方針
（自作エンジン＋自作 ONNX ランタイム、Python は速いだけで必須ではない、パリティはテストで縛る）。

## なぜこの順番か（手法の比較）

「人の位置がわかる群衆検知」の最良は P2PNet 系だが、**FIDTM は CSRNet と出力の形が同じ**（1 チャンネル
のマップ）なので、CSRNet を作れば追加コストがほぼ無い。だから CSRNet → FIDTM → 必要なら P2PNet の順。

| 手法 | 出すもの | 位置精度 | 実装コスト（このリポの都合） |
|---|---|---|---|
| **CSRNet**（CVPR 2018） | 密度マップのみ | 位置は出ない | 低。ただし **dilated conv が必須**（自作エンジンに無かった → 追加した） |
| **FIDTM**（2021-22） | 1ch マップ → 局所最大で点 | 密集でも点が分離する。閾値と NMS 半径に依存 | **CSRNet の配管を再利用**＋局所最大の後処理 |
| P2PNet（ICCV 2021） | 点を直接回帰 | 厳しい距離閾値では FIDTM より強い。後処理のヒューリスティクスが無い | Hungarian マッチングが学習ループに入る（両言語＋パリティ） |
| CLTR / STEERER / APGCC（22-24） | 点 | さらに上 | Transformer・多スケール分離で重い |
| 頭の box を YOLO で検出 | box | 密集しすぎなければ実用十分 | 姉妹リポの一式がそのまま使える |

判断基準は先に決めておく: **数え上げは MAE/MSE、位置は距離閾値つきの F1**。FIDTM の F1 が足りなければ
P2PNet に進む。

## いま動くもの

```sh
sh build/gcc.sh pure/crowd.cpp -o crowd.exe
./crowd.exe init-csrnet --out models/csrnet.onnx --imgsz 384       # グラフを C++ が書く
./crowd.exe infer --img <画像> --model models/csrnet.onnx          # 自作ランタイムで推論
sh build/gcc.sh pure/gradcheck.cpp -o gradcheck.exe && ./gradcheck.exe
```

| 検証 | 結果 |
|---|---|
| dilated conv の勾配 | dilation 1/2/3 で解析勾配と中心差分が **3e-04 以下**一致（`pure/gradcheck.cpp`） |
| CSRNet のパラメータ数 | **16,263,489** = 論文の 16.26M と一致 |
| ONNX の妥当性 | `onnx.checker` PASS、onnxruntime で実行可（`[1,3,384,384]` → density `[1,1,48,48]`） |
| 自作ランタイム ⇔ onnxruntime | 同じ画像で count **338.27 / 338.2708**、max 0.1532 / 0.153233 |

（この count に意味は無い。ランダム初期化だと ReLU が大半死んでマップがほぼ一定になる ＝ マップの分散
1.7e-06。CSRNet が VGG-16 の**事前学習**前段を要求する理由がそのまま出ている。）

## 構造（CSRNet）

```
入力 [1,3,S,S]
  │
  ├ 前段: VGG-16 の最初の 10 conv（conv1_1..conv4_3）＋ 2x2 pool 3 回     → S/8
  │        VGG-16 の 4 番目の pool は**落とす**（これが 1/8 を保つ仕掛け）
  │
  ├ 後段: 3x3 conv × 6、**すべて dilation 2**（512,512,512,256,128,64）  → S/8 のまま
  │        受容野だけ広げて解像度を落とさない
  │
  └ 1x1 conv → 1ch 密度マップ [1,1,S/8,S/8]   合計が人数
```

パラメータ名は torchvision の VGG-16 の state_dict 名（`features.0.weight` …）にしてあるので、
`--from-pt vgg16.pth` がテンソル単位で載る。後段は `backend.<n>.weight`。

## 対等性（yolo_lpr_cpp と同じ規律）

| 機能 | Python (`tools/`) | C++ (`pure/`) | パリティの条件 |
|---|---|---|---|
| グラフ生成 | 予定 | `crowd init-csrnet` ✅ | 同じ ONNX が出る（重み以外バイト一致） |
| 推論 | 予定 | `crowd infer` ✅ | 同一画像で count が一致（対 onnxruntime で実測 4e-04 相対） |
| 密度ラベル生成 | 予定 | 予定 | 同じ点群から同じマップ（バイト一致） |
| 学習 | 予定 | 予定 | 同じ seed・同じ batch で step1 の loss が一致 |
| 評価（MAE / F1） | 予定 | 予定 | 同じデータで同じ数値 |

## ライセンス

自前コードは BSD-3-Clause。VGG-16 の事前学習重みは torchvision（BSD-3-Clause）由来で、
リポジトリには**含めない**（各自 `--from-pt` で渡す）。データセットも同様に含めない。

`models/*.onnx` も git に入れない: CSRNet は 16.3M パラメータ ＝ 1 個 **62MB** あり、
ランダム初期化のものは `crowd init-csrnet` の 1 コマンドで作り直せる。
学習済みのもの（特に軽量版）は、デモに必要になった時点で例外として入れる。
