# models/

生成物は基本 gitignore してある（`crowd init-csrnet` の 1 コマンドで作り直せるので）。
例外は**学習済みで、そのまま使える重みだけ**。いま入っているのは 2 本。

| | パラメータ | サイズ | F1 σ8 / σ4（test 316 枚） | WASM で 512x336 | 1024x768 |
|---|---|---|---|---|---|
| `fidt_light_B.onnx` | 1,022,329 | **4.1MB** | **0.7070** / 0.4683 | **2.9 秒** | 10.8 秒で通る |
| `fidt_partB.onnx` | 16,318,857 | 62MB | **0.7971** / 0.5695 | 21.5 秒 | 2GB 上限で落ちる |
| （論文 FIDTM, HRNet-W48） | 約 65M | — | 0.776 / 0.586 | — | — |

**ブラウザに載せるなら軽量版**（1/15 の大きさ、7.4 倍速く、全解像度が通る。F1 は 0.09 落ちる)。
軽量版は幅 0.25 で VGG-16 の転移が載らないので**ゼロから 24000 step**学習した（`--width 0.25`）。

## `fidt_light_B.onnx` — 軽量版（ブラウザ用の既定）

`crowd init-csrnet --decoder 4 --width 0.25` のグラフを、参考実装のレシピ（256 クロップ・
batch 16・Adam lr 1e-4・wd 5e-4）で **24000 step ゼロから**学習したもの。入出力の形と
座標の出し方は下の通常版と同じ（**python 側で読むときは `--width 0.25` が必要**）。

* test 316 枚で **F1 0.7070**（σ8）/ 0.4683（σ4）、マップ最大 0.483
* 24000 step でもまだ上昇中だった（18000: 0.688 → 22000: 0.704 → 24000: 0.715、eval 60 枚）

## `fidt_partB.onnx` — FIDTM（位置推定）、ShanghaiTech Part B

WASM デモ用。**入力の 1/2 解像度**で FIDT マップを出し、その極大が頭の位置になる。

| | |
|---|---|
| 構成 | CSRNet（VGG-16 前段 + dilated conv 6 本）+ デコーダ 2 段（`crowd init-csrnet --decoder 4`） |
| パラメータ | 16,318,857（62MB。**軽量版は未着手** — RESUME の M8） |
| 入力 | `input` = `[1,3,H,W]` float32、H/W は動的。**ImageNet 正規化**（mean 0.485/0.456/0.406、std 0.229/0.224/0.225、RGB、0-1 に割ってから） |
| 出力 | `[1,1,H/2,W/2]` float32。活性化なし。値は 0..1 付近（学習済みのこの重みは最大 0.63 前後） |
| 学習 | Part B train 400 枚、256 クロップ・batch 16・Adam lr 1e-4 固定・weight decay 5e-4、22,000 step（best） |
| 精度（test 316 枚） | **F1 0.7971**（距離閾値 8px、precision 0.876 / recall 0.732）、**F1 0.5695**（4px）。論文 FIDTM は 0.776 / 0.586 |

### 座標の出し方（参考実装と同じ LMDS）

1. 3x3 の窓で局所最大を取る（半径 1、同値は走査順で先着）
2. **そのマップ自身の最大値の 100/255 倍**（≈0.392 倍）未満は捨てる ← **絶対値ではない**
3. マップの最大値が 0.1 未満なら、何も検出しない（負例ガード）
4. 残った点の座標を **2 倍**すると入力画像の座標になる

`pure/density.hpp` の `den::lmds()` と `tools/density.py` の `D.lmds()` が同じことをする。
絶対閾値（例えば 0.5）で切ると、同じ重みでも F1 が 0.005 まで落ちる。理由は RESUME に書いてある。

### 測り直す

```sh
# Part B の test セットを scratch/sht/part_B に置いてから（RESUME「環境とデータの取り方」）
python tools/eval.py --data scratch/sht/part_B --model models/fidt_partB.onnx --fidt --down 2
./crowd.exe eval --data scratch/sht/part_B --model models/fidt_partB.onnx --fidt --down 2
```

### 作り直す

```sh
./crowd init-csrnet --out models/fidt.onnx --decoder 4 --from-pt vgg16_front.pth
python tools/train_csrnet.py --data <part>/part_B --init models/fidt.onnx --fidt --down 2 \
    --crop 256 --batch 16 --lr 1e-4 --adam-wd 5e-4 --steps 24000 --eval-every 2000 \
    --eval-limit 60 --log run.csv --export models/fidt_partB.onnx --ckpt ck/fidt.ck --ckpt-every 500
```
