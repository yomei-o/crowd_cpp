# crowd_cpp — 群衆の数え上げと人の位置推定を C++ と Python の両方で

密度マップ回帰（**CSRNet**）から始めて、人の位置が出せる方式（**FIDTM**）まで。**学習も推論も評価も
両言語**ででき、成果物は全段 ONNX。姉妹リポの [yolo_lpr_cpp](https://github.com/yomei-o/yolo_lpr_cpp)
と同じ設計方針（自作エンジン＋自作 ONNX ランタイム、Python は速いだけで必須ではない、
パリティはテストで縛る）。

進捗・実測値・落とし穴・次の一手は **[RESUME.md](RESUME.md)** に集約してある。

## なぜこの順番か（手法の比較）

「人の位置がわかる群衆検知」の最良は P2PNet 系だが、**FIDTM は CSRNet と出力の形が同じ**（1 チャンネル
のマップ）なので、CSRNet を作れば追加コストがほぼ無い。だから CSRNet → FIDTM → 必要なら P2PNet の順。

| 手法 | 出すもの | 位置精度 | 実装コスト（このリポの都合） |
|---|---|---|---|
| **CSRNet**（CVPR 2018） | 密度マップのみ | 位置は出ない | 低。ただし **dilated conv が必須**（自作エンジンに無かった → 追加した） |
| **FIDTM**（2021-22） | 1ch マップ → 局所最大で点 | 密集でも点が分離する。出力解像度が天井を決める（下記） | **CSRNet の配管を再利用**＋デコーダ＋局所最大 |
| P2PNet（ICCV 2021） | 点を直接回帰 | 厳しい距離閾値では FIDTM より強い | Hungarian マッチングが学習ループに入る |
| CLTR / STEERER / APGCC | 点 | さらに上 | Transformer・多スケール分離で重い |
| 頭の box を YOLO で検出 | box | 密集しすぎなければ実用十分 | 姉妹リポの一式がそのまま使える |

判断基準は先に決めてある: **数え上げは MAE/RMSE、位置は距離閾値つきの F1**。

## いま動くもの

```sh
sh build/gcc.sh pure/crowd.cpp -o crowd.exe
EXTRA="-DUSE_EIGEN" sh build/gcc.sh pure/crowd.cpp -o crowd.exe   # 学習が 10.7 倍速い（実測）
sh build/gcc.sh pure/gradcheck.cpp -o gradcheck.exe && ./gradcheck.exe   # dilated conv の勾配

# 出発点のモデルを C++ が書く（VGG-16 前段の転移つき。--decoder 4 で FIDTM 用の 1/2 出力）
./crowd.exe init-csrnet --out models/csrnet.onnx [--from-pt vgg16_front.pth] [--decoder 0|2|4]

# ラベル（.mat を純 C++ で読む。密度なら合計＝人数、--fidt なら極大＝頭の位置）
./crowd.exe labels --mat GT_IMG_1.mat --img IMG_1.jpg [--adaptive | --fidt]

# 学習（丸ごと1枚 batch 1 が既定＝論文と同じ。--optim sgd で参照実装のレシピ）
./crowd.exe train --data <ShanghaiTech>/part_B --init models/csrnet.onnx --steps 1000 \
    --eval-every 200 --export models/out.onnx

# 長い学習は中断される前提で回す（Kaggle のセッションは実測 2 時間前後で落ちる）
./crowd.exe train ... --log run.csv --ckpt run.ck --ckpt-every 500   # 途中経過と再開点を残す
./crowd.exe train ... --log run.csv --resume run.ck                  # 落ちた続きから

# 評価（学習を回さずに測る。--sweep でピーク閾値を振って F1 と振幅を出す）
./crowd.exe eval --data <ShanghaiTech>/part_B --model models/out.onnx           # 密度: MAE/RMSE
./crowd.exe eval --data <ShanghaiTech>/part_B --model models/fidt_B.onnx --fidt --down 2 --sweep

# 推論（自作ランタイム。任意サイズを受ける）
./crowd.exe infer --img <画像> --model models/csrnet.onnx
```

Python 側は `tools/csrnet.py`（ONNX を名前で読み書きする torch 実装。`--decoder 2|4` の
デコーダも同じテンソル名で持つ）、`tools/train_csrnet.py`（学習）、`tools/eval.py`（評価。
`crowd eval` と**出力が一字一句同じ**）。`--fidt` で FIDT 目標と F1 評価に切り替わる。

## ブラウザで動かす（WASM）

```sh
sh build/emcc.sh wasm/crowd_wasm.cpp -o wasm/crowd.js
node wasm/test_node.js                       # 実画像で onnxruntime と数値が一致するか
python -m http.server -d . 8000              # http://127.0.0.1:8000/wasm/
```

推論ライブラリは使わず、`pure/onnx_run.hpp` をそのまま emcc でビルドしている。画像かカメラ 1 枚を
入れると、頭の位置（青丸）とヒートマップと人数が出る。通常版は **512x384 で 1 枚 25 秒**、
1024x768 は wasm32 の 2GB 上限に当たって動かない（中間テンソルを全部保持する実装のため）。
詳細と直し方は [RESUME.md](RESUME.md) の「WASM デモ」。

## 学習済みの重み

`models/fidt_partB.onnx`（62MB）を 1 本だけリポジトリに入れてある。FIDTM の位置推定を
ShanghaiTech Part B で学習したもの（**test 316 枚で F1 0.7971 / 8px**）。入力の正規化・出力の
読み方・極大の取り方（**閾値は絶対値ではなくマップ最大値の 100/255 倍**）は
[models/README.md](models/README.md) に書いてある。ブラウザに 62MB は重いので、
軽量版（M8、`--width 0.25`）ができたら差し替える。

## 検証済みの数字

| 検証 | 結果 |
|---|---|
| dilated conv の勾配 | dilation 1/2/3 で解析勾配と中心差分が **3e-04 以下** |
| CSRNet のパラメータ数 | **16,263,489** = 論文の 16.26M |
| ONNX の妥当性 | `onnx.checker` PASS、H/W 動的宣言で任意サイズ（ORT で 384 / 768x1024 / 664x1000） |
| forward 三者一致 | 自作 **338.27** / onnxruntime **338.2703** / torch **338.2703** |
| VGG-16 前段の転移 | conv4_3 の活性が torchvision と**相対 7.9e-07**（純 C++ の `.pt` リーダ経由） |
| dilated conv 6 本通過後（自作 ⇔ ORT） | count -2028.06 / -2028.0618 |
| ラベル生成 C++ ⇔ Python | 3 種すべて相対 **5e-06 以下**、密度の合計は点数と 2e-06、FIDT のピーク 907/907 |
| **学習 C++ ⇔ Python（同じバッチ）** | loss **完全一致 3.453135**、勾配 34 テンソルの最悪 **2.57e-05** |
| **CSRNet の精度（ShanghaiTech Part B）** | 20000 step で **test MAE 33.10 / RMSE 45.21**（316 枚。論文 10.6 だが予算は参考実装の 1/24） |
| **推論の両言語一致（学習済み CSRNet）** | 自作 C++ ランタイム **MAE 41.25** / torch **41.32**（同じ 20 枚） |
| **FIDTM の位置推定（Part B、test 316 枚）** | 参考実装の LMDS で **σ8 0.7971 / σ4 0.5695**（論文 FIDTM は 0.776 / 0.586。**σ8 は +0.021 で上回った**。予算は論文の 1/3 弱、バックボーンは CSRNet/VGG-16） |
| 出力ストライドの上限（`eval --labels`） | σ8/σ4 で 1/8 → 0.850/0.701、1/4 → 0.956/0.956、1/2 → **0.994/0.994**、1/1 → 0.9996 |
| **`--resume`（両実装）** | 途中で kill して再開すると、以降の loss が無停止運転と**印字桁まで一致**（`tools/parity/resume.py` が py / cpp 両方で PASS） |
| デコーダ付きグラフ（1/2 出力）torch ⇔ ORT | 38 テンソル全部が名前と形で一致、出力は**相対 4.3e-07** |
| `crowd eval` ⇔ `tools/eval.py` | `--labels`（ネットワークを通さない）は**完全一致**。モデルを通す数字は実データ 6 枚で F1 が **0.004 以内**、合成データで 0.0002 以内（極大の数え上げは離散なので、1 セル 4e-7 の差が境目のピークを入れ替える）。`tools/parity/eval.py` が縛る |
| ラベル表現の天井（位置） | 1/8 で F1 0.737、1/4 で 0.933、**1/2 で 0.981**（precision はどれも 1.000） |
| CSRNet の精度（Part B） | 8000 step で MAE **43.6**（まだ下降中、セッション落ちで中断）。**論文は 10.6** — 差は予算（[RESUME](RESUME.md) に切り分け） |

## 構造（CSRNet）

```
入力 [1,3,H,W]（H,W は任意）
  │
  ├ 前段: VGG-16 の最初の 10 conv（conv1_1..conv4_3）＋ 2x2 pool 3 回     → 1/8
  │        VGG-16 の 4 番目の pool は**落とす**（これが 1/8 を保つ仕掛け）
  │
  ├ 後段: 3x3 conv × 6、**すべて dilation 2**（512,512,512,256,128,64）  → 1/8 のまま
  │        受容野だけ広げて解像度を落とさない
  │
  ├（FIDTM 用）デコーダ: nearest x2 ＋ 3x3 conv を 1〜3 段              → 1/4 か 1/2
  │
  └ 1x1 conv → 1ch マップ   密度なら合計が人数、FIDT なら極大が頭の位置
```

パラメータ名は torchvision の VGG-16 の state_dict 名（`features.0.weight` …）なので
`--from-pt vgg16.pth` がテンソル単位で載る。後段は `backend.<n>`、デコーダは `decoder.<n>`。

## 対等性（yolo_lpr_cpp と同じ規律）

| 機能 | Python (`tools/`) | C++ (`pure/`) | パリティの条件と実測 |
|---|---|---|---|
| グラフ生成 | — | `crowd init-csrnet` ✅ | ORT と自作ランタイムの両方が読める |
| `.pt` からの転移 | `torch.load` | `pure/ptio.hpp` ✅ | 前段の活性が torchvision と 7.9e-07 |
| `.mat` 読み込み | `scipy.io.loadmat` | `pure/matio.hpp`（zlib 対応） ✅ | 同じ座標 |
| ラベル生成（密度・FIDT） | `tools/density.py` ✅ | `pure/density.hpp` ✅ | 相対 5e-06 以下、ピーク数は完全一致 |
| 学習 | `tools/train_csrnet.py` ✅ | `crowd train` ✅ | 同じバッチで loss 完全一致、勾配 2.57e-05 |
| 評価（MAE / F1） | ✅ | ✅（`--fidt` で F1 に切替） | 同じデータで同じ数値 |
| 推論 | `csrnet.py` + ORT ✅ | `crowd infer` ✅ | forward 三者一致 |

テストは `tools/parity/`（`labels.py`, `train.py`, `vgg_front.py`）と `pure/gradcheck.cpp`。

## ライセンス

自前コードは BSD-3-Clause。`pure/ptio.hpp` は姉妹リポ yolov8_cpp から移植（同じ作者・BSD-3-Clause）。
VGG-16 の事前学習重みは torchvision（BSD-3-Clause）由来で、リポジトリには**含めない**
（各自 `--from-pt` で渡す）。データセットも同様に含めない。

`models/*.onnx` も git に入れない: CSRNet は 16.3M パラメータ ＝ 1 個 **62MB** あり、
ランダム初期化のものは `crowd init-csrnet` の 1 コマンドで作り直せる。
学習済みのもの（特に軽量版）は、デモに必要になった時点で例外として入れる。
