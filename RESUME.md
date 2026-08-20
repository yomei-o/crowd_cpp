# RESUME — crowd_cpp の進捗と残作業

方針は [README.md](README.md)。ここは**現在地・次の一手・実測した知見・落とし穴**を書き続けるファイル。
引き継ぐ人はこのファイルを上から読めば足りるようにしてある。

## 現在地（2026-08-20）

**CSRNet の配管が両言語で完成し、パリティで縛られている。** 精度はまだ論文値に遠く、その理由は
実装ではなく学習予算だと切り分け済み（下の「なぜ MAE が論文値に届かないか」）。
FIDTM（位置推定）は部品が全部揃って、学習だけが残っている。

| 検証 | 結果 |
|---|---|
| dilated conv の勾配（自作エンジンに追加した） | dilation 1/2/3 で解析勾配と中心差分が **3e-04 以下**一致（`pure/gradcheck.cpp`） |
| CSRNet のパラメータ数 | **16,263,489** = 論文の 16.26M |
| ONNX の妥当性 | `onnx.checker` PASS、onnxruntime で実行可。H/W は動的宣言なので任意サイズ |
| forward 三者一致 | 自作ランタイム **338.27** / onnxruntime **338.2703** / torch **338.2703** |
| VGG-16 前段の転移（純 C++ の `.pt` リーダ） | conv4_3 の活性が torchvision と**相対 7.9e-07** |
| dilated conv 6 本通過後（自作 ⇔ ORT） | count -2028.06 / -2028.0618、max -0.8683 / -0.868379 |
| `.mat` リーダ（zlib 圧縮つき） | 本物の ShanghaiTech GT を読める（GT_IMG_1 = 1546 点） |
| ラベル生成 C++ ⇔ Python | 密度2種＋FIDT すべて相対 **5e-06 以下**、密度の合計は点数と 2e-06、FIDT のピーク 907/907 |
| **学習 C++ ⇔ Python（同じバッチ）** | loss **完全一致 3.453135**、勾配 34 テンソルの最悪 **2.57e-05** |
| CSRNet の精度（ShanghaiTech Part B） | 3000 step で test MAE **72.3**（論文 10.6）。20000 step を実行中 |

## 次の一手（優先順）

1. **20000 step の学習をやり直す**（前回はセッション落ちで step 8000 で消えた。下の「学習の記録」）。
   有望なのは **cosine（`--lr 2e-6 --lr-final 0.02`）** で、8000 step で MAE 43.6・まだ下降中だった。
   **eval ごとにモデルを引き取ること**（落とし穴の表を見よ）。完走したら自作 C++ ランタイムでも
   同じ MAE が出るかを確認する（**ここまでやれば M6 完了**）。
   ```sh
   python tools/train_csrnet.py --data ../sht/ShanghaiTech/part_B --init models/csrnet_vgg.onnx        --steps 20000 --lr 2e-6 --lr-final 0.02 --eval-every 2000 --eval-limit 120        --log run.csv --export models/csrnet_B.onnx
   ```
2. **FIDTM の学習（M7）**。部品は全部ある:
   ```sh
   ./crowd init-csrnet --out models/fidt.onnx --decoder 4 --from-pt vgg16_front.pth   # 1/2 出力
   python tools/train_csrnet.py --data <part>/part_B --init models/fidt.onnx --fidt --down 2 \
       --crop 256 --batch 8 --lr 1e-5 --steps 8000 --eval-every 1000 --export models/fidt_B.onnx
   ```
   **FIDTM はクロップ学習・batch 8 が使える**（指標が局所的なので合計を守る必要がない）。CSRNet が
   丸ごと1枚 batch 1 に縛られたのとは逆で、予算に対する効率は良いはず。これは予想なので測って確かめる。
   期待値の基準はラベル側の天井（1/2 で F1 0.981）。
3. ~~C++ 側にも FIDT 学習と F1 評価を入れる~~ → **完了**（2026-08-20）。
   `crowd train --fidt --down 2 --loc-thr 8` で FIDT を学習し、評価が F1 に切り替わる
   （デコーダ込み 38 テンソルが学習対象、未学習の初期値は F1 0.0000 ＝ 期待どおり）。
4. **軽量版（M8）**。`--width 0.25` あたりで学習し、WASM に載せる。CSRNet は 16M パラメータ = 62MB
   あるので、ブラウザに載せるには軽量化が前提。
5. P2PNet（M9）は FIDTM の F1 を見てから判断する。

## マイルストーン

- [x] **M0 足場とエンジン** — `pure/` を [yolo_lpr_cpp](https://github.com/yomei-o/yolo_lpr_cpp) から移植。
- [x] **M1 dilated convolution** — `conv2d(..., dil)` と ONNX の `dilations`。勾配は `pure/gradcheck.cpp` で縛る。
- [x] **M2 CSRNet のグラフ生成（C++）** — `crowd init-csrnet`。16.26M パラメータ、ORT 検証済み。
- [x] **M3 VGG-16 転移の検証** — `--from-pt` で前段 20 テンソル、torchvision と相対 7.9e-07。
- [x] **M4 データとラベル** — `.mat` リーダ（zlib 対応）、密度（固定σ/適応σ）と FIDT、両言語パリティ。
- [x] **M5 学習（両言語）** — `crowd train` と `tools/train_csrnet.py`、同じバッチで loss 完全一致。
- [~] **M6 評価** — 丸ごと1枚の MAE/RMSE は両言語にある。**論文値との比較が残り**（実行中）。
- [~] **M7 FIDTM** — ラベル・デコーダ（`--decoder 2|4`）・極大検出・F1 指標・**両言語の学習**まで完了。
      残りは実際の学習と F1 の測定だけ。
- [ ] **M8 軽量版と WASM** — `--width` で細くして学習、ブラウザに載せる。
- [ ] **M9 P2PNet** — M7 の F1 が足りなければ。Hungarian マッチングを両言語に。

## なぜ MAE が論文値に届かないか — 切り分けの記録（2026-08-20）

参照実装 [leeyeehoo/CSRNet-pytorch](https://github.com/leeyeehoo/CSRNet-pytorch)（論文第一著者）を読んで
突き合わせた結果、**違いはハイパラ 2 点だけ**だった:

| | 参照実装 | こちら |
|---|---|---|
| ラベル | 全解像度の密度 → 1/8 に resize ×64（合計＝人数） | 1/8 で直接生成（合計＝人数） **同じ** |
| 損失 | `MSELoss(size_average=False)` = 合計 | 合計 **同じ** |
| 正規化 | ImageNet mean/std | **同じ** |
| 入力 | batch 1・丸ごと1枚（クロップ分岐は `if False:` で無効） | **同じ**（下記の経緯で修正した） |
| 最適化 | **SGD 1e-7 固定**, momentum 0.95, wd 5e-4 | Adam 1e-5 → 1e-6 |
| 総ステップ | 300枚 × **4回/epoch** × 400 epoch = **約48万** | 3,000 → 20,000 |

参照の `adjust_learning_rate` は `scales=[1,1,1,1]` なので**減衰すらしていない**。
つまり「lr を 100 分の 1 にして 160 倍長く回す」のが正体で、T4 実測 0.485 秒/step だと
**48 万 step は約 65 時間**。Kaggle の 1 セッション上限（9 時間）でも 6.7 万 step が上限。

### 途中で潰した 2 つの間違い（どちらも測って分かった）

**(1) 384 クロップで学習していた** — loss は 3.4 → 1.2 と順調なのに丸ごと1枚の MAE が
50 → 158 → 57 → 92 と暴れた。推測せず 1 枚ずつ調べたら:

| 画像 | 正解 | 予測 | map 平均 | map 最小 |
|---|---|---|---|---|
| IMG_1 | 23 | **135.6** | 0.0110 | -0.0465 |
| IMG_103 | 57 | **159.6** | 0.0130 | -0.1273 |

IMG_1 は `0.0110 × 12288 セル = 135` で、**予測のほぼ全部が一定の DC 成分**だった。
Part B のクロップはどれも人を含むので「どこでも一定値を出す」がクロップ損失の良い局所最適になり、
面積 5 倍の全画像では破綻する。→ 既定を丸ごと1枚 batch 1（参照と同じ）に変更。

**(2) 学習率が大きすぎた** — 丸ごと1枚にしても MAE は 151 → 60 → 88 → 347 → 121 と振れ、
しかも **train MAE ≈ test MAE**（過学習でもプロトコル不一致でもない＝収束していない）。
2 つの仮説を GPU 2 枚で同時に検証した:

| 3000 step | 結果 |
|---|---|
| Adam **1e-6** | 302 → 192 → 193 → 146 → 79 → **72.3**（単調に下降、まだ収束前） |
| Adam 1e-5 ＋ count-weight | 200 → 108 → 131 → 122 → 251 → 185（振動のまま） |

→ **lr が主因**で、count-weight（合計を損失で直接拘束する項）は効かなかった。
count-weight は外れた仮説として `--count-weight`（既定オフ）に残してある。

### この問題の構造（覚えておく価値がある）

密度マップの**合計**は、一様な DC 偏りに極端に敏感である:
1 セルあたり +0.01 の偏りは 12288 セルで **+123 人**の誤差になるが、合計二乗誤差は **+1.2** しか増えない。
つまり**報告している指標が、学習している損失にほとんど拘束されていない**。
これが「loss は下がるのに MAE が 5 倍振れる」の正体で、参照実装が lr 1e-7 × 48 万 step という
極端に慎重な設定を採っている理由でもある。FIDTM ではこの問題が消える（下記）。

## FIDTM の位置推定は出力解像度で天井が決まる — 2026-08-20 実測

**学習の前に分かる話**なので先に測った。1024x768 に 1546 点（Part A の密な側）の**正解の** FIDT
マップから局所最大を取り、距離 8px で正解と greedy マッチした結果:

| 出力 down | マップ | ピーク | recall | **F1** | precision |
|---|---|---|---|---|---|
| 8（CSRNet と同じ） | 128x96 | 902 | 0.583 | **0.737** | **1.000** |
| 4 | 256x192 | 1352 | 0.875 | 0.933 | **1.000** |
| 2 | 512x384 | 1489 | 0.963 | **0.981** | **1.000** |

precision が全部 1.000 なのが重要で、1/8 でも「出したピークは必ず 8px 以内」＝誤検出ではなく
**融合による取りこぼし**が唯一の損失源。CSRNet の 1/8 は数え上げには十分でも位置推定には構造的に
足りない（FIDTM 論文が全解像度で出す理由）。

実装済み: `crowd init-csrnet --decoder 0|2|4`（1/8・1/4・1/2）。**パラメータ増は +0.3% だけ**
（16,263,489 → 16,318,857）＝デコーダは安く、支配的なのは VGG 前段。

**FIDTM が CSRNet より予算に優しいと予想する根拠**（要検証）:

| | CSRNet（密度） | FIDTM |
|---|---|---|
| 指標 | マップの**合計**＝人数 | 局所最大の位置（F1） |
| 一様な DC 偏り +0.01 | **+123 人**の誤差、loss は +1.2 | 極大の位置はほぼ動かない |
| 目標値の範囲 | 混雑度で変わる（数十〜数千） | **[0,1] に有界** |
| 学習単位 | 合計を守るため丸ごと1枚 batch 1 | 局所指標なので**クロップ batch 8 が使える** |

## 学習の記録

| 日付 | 設定 | 結果 |
|---|---|---|
| 08-20 | Part B, 384 クロップ, Adam 1e-5, 2000 step | best MAE 50.61（DC 成分を学習していた） |
| 08-20 | Part B, 丸ごと1枚, Adam 1e-5, 3000 step | best MAE 60.48（振動） |
| 08-20 | Part B, 丸ごと1枚, Adam 1e-6, 3000 step | best MAE **72.3**（単調下降、収束前） |
| 08-20 | Part B, 丸ごと1枚, Adam 1e-5 + count-weight 0.001, 3000 step | best MAE 108.1（振動） |
| 08-20 | Part B, 丸ごと1枚, Adam 1e-6, 20000 step 予定 | **step 8000 で MAE 58.5**（2000:145.6 / 4000:153.5 / 6000:71.4）。Kaggle セッションが落ちて中断 |
| 08-20 | Part B, 丸ごと1枚, Adam 2e-6→cosine 4e-8, 20000 step 予定 | **step 8000 で MAE 43.6**（train 38.6。2000:213.6 / 4000:69.9 / 6000:68.9）。同上で中断 |

**中断の経緯と教訓**: 20000 step の 2 本を走らせている途中（step 8000 過ぎ、経過 1 時間）に
Kaggle セッションが 502 で落ち、プロセスとコンテナ上の best モデル（`models/csrnet_B_*.onnx`）と
CSV ログを失った。**改善する eval のたびに書き出していたのだから、その都度ダウンロードしておくべき
だった**（同じ日にすでに 1 回セッションが切れているのを見ていたのに、完走前提で組んでいた）。
次に長い学習を回すときは、eval ごとに `curl "$KB/download?path=...&raw=1"` で引き取る。

**残っている情報からの見立て**: cosine（2e-6 から減衰）の方が明確に速く、8000 step で 43.6、
まだ train/test が近く下降中だった。20000 step まで回せば 30 前後は見込めそうだが、**論文の 10.6 には
届かない**予算であることは変わらない（参照実装は 48 万 step）。

## 環境とデータの取り方

**データ（ShanghaiTech）**: Kaggle の `tthien/shanghaitech`（349MB）。**Kaggle ノートブックの中では
`kaggle` CLI の認証が通っている**ので、そのまま落とせる（5.7 秒）:

```sh
mkdir -p sht && cd sht && kaggle datasets download -d tthien/shanghaitech -p . --unzip
# -> sht/ShanghaiTech/part_A|part_B/{train,test}_data/{images,ground-truth}
#    part_A 300/182、part_B 400/316 の正規の split
```

**VGG-16 の重み**: torchvision から取って前段だけ保存し、C++ の `.pt` リーダに渡す:

```sh
python -c "
import torch, torchvision
m = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
sd = {k: v for k, v in m.state_dict().items() if k.startswith('features.') and int(k.split('.')[1]) <= 21}
torch.save(sd, 'vgg16_front.pth')"
./crowd init-csrnet --out models/csrnet_vgg.onnx --from-pt vgg16_front.pth
```

**GPU**: 姉妹リポの `kaggle_server_cpp`（kbridge）経由で Kaggle の T4 x2 を使う。手順は
あちらの `FOR_AGENTS.md`。**"VSCode Compatible URL" はそれ自体が認証情報**なので、ログにも
コミットにも出さない（kbridge は `****` にマスクする）。セッションは切れるので都度人にもらう。
`scratch/kb.py`（短いコマンド）と `scratch/kb_job.py`（長いジョブ）と `scratch/wait_job.py` がある。

## 速度 — CPU で本番学習は無理、という数字（2026-08-20 実測）

| 経路 | 入力 | 1 step | 換算（768x1024） |
|---|---|---|---|
| Python / torch / **T4** | 768x1024 丸ごと1枚 | **0.46 秒** | 0.46 秒 |
| C++ / 自作エンジン / CPU 4 コア | 512x384 丸ごと1枚 | 126 秒 | 約 500 秒 |
| C++ / **`-DUSE_EIGEN`** / CPU 4 コア | 512x384 丸ごと1枚 | **11.75 秒** | 約 47 秒 |

いずれも「1 step の実行」と「4 step の実行」の差分から 3 step 分を出した値（起動と ONNX 読み込みと
ラベル生成を除ける）。

**`-DUSE_EIGEN` は 10.7 倍速い。** 姉妹リポでは 2.5〜3.4 倍だったので、そこより大きく効く。
CSRNet は大きくて密な conv ばかりなので、GEMM の質がそのまま出る形になっている。
`build/gcc.sh` は `pure/third_party/eigen_flat` を include するので `EXTRA="-DUSE_EIGEN"` だけで通る。

それでも GPU の約 100 倍なので、20000 step は 261 時間で非現実的。本番学習は Python/GPU、C++ は
「同じ結果に到達できることの証明」という分担にしている（姉妹リポと同じ立場）。
ただし**検証用の小規模実行（数十〜数百 step）は Eigen ビルドで実用的**なので、C++ 側の学習が
本当に前進するかを確かめるときはこれを使う。

step 1 の loss 9596 が両言語で説明できることは確認した: VGG 初期化直後のマップは 1 セル約 -0.88 で、
0.88^2 x 12288 セル ≈ 9500。つまり初期損失のほぼ全部が DC 成分。

## 落とし穴（実際に踏んだもの。再発させない）

| 事象 | 症状 | 対策 |
|---|---|---|
| `.mat` の最終要素はパディングされない | 要素長を「パディング込みで親に収まるか」で検査していたため、末尾要素を弾いて「変数なし」になる（128+8+133=269 バイトの圧縮ファイル） | payload だけを検査する |
| MATLAB v7 は既定で zlib 圧縮 | inflate が無いと本物の GT が 1 つも読めない | stb_image 同梱の zlib を使う（依存は増えない） |
| stb は実装がインクルードガードの外 | ヘッダから include すると、実装を定義した .cpp が壊れる | **stb は .cpp だけ**。ヘッダでは関数を `extern "C"` 宣言する |
| `models/` が git に無い | `.gitignore` で生成 ONNX を除外＋git は空ディレクトリを持たない → clone 直後の保存が黙って失敗し、症状は数手あとの「ファイルが無い」だけ | 書き込み前に mkdir -p（`make_parent`） |
| gradcheck を出力全体の重み付き和で作る | float32 の差分ノイズが相対 5e-3 出て、**正しいコードでも** dilation 1 で FAIL する | 出力 1 要素だけを微分し、勾配が極小（<1e-2）の入力は比較から外す |
| 生成 ONNX を commit | CSRNet は 1 個 **62MB**。GitHub が 50MB 超で警告 | `models/*.onnx` は gitignore。1 コマンドで作り直せる |
| 学習の途中経過が見えない | `... \| tail -45` はバッファされるので、走っている間ログが空になる | `--log <csv>`（毎行 flush）を使う。`grep -E 'eval @'` は完了後にしか出ない |
| **Kaggle セッションは落ちる** | 20000 step を 2 本、1 時間走らせたところで 502 になり、プロセスと書き出し済みモデルと CSV を失った（step 8000 の数字だけが会話に残った） | **成果物は書かれた時点で引き取る**。eval ごとに `curl "$KB/download?path=<repo>/models/x.onnx&raw=1" -o` する。9 時間の上限より先に落ちる前提で組む |

**作業上の注意**（このリポジトリを編集するとき）: Bash の heredoc は `\n` を実改行に化けさせる。
C++/Python の文字列リテラルを含むパッチは Write/Edit ツールを使うか、`chr(92)+'n'` を使う。
実際に何度も壊してから気付いた（症状は「文字列リテラルが閉じていない」コンパイルエラー、
Python なら docstring が 1 行に潰れる）。

## 未確認事項

1. ~~入力サイズとメモリ~~ → 測った（下の「速度」節）。
2. **適応ガウシアンの kNN の同順位**: 両言語で `partial_sort` / `np.sort` を使っていて実測は一致
   しているが、完全に同距離の点が複数ある場合の順序は保証していない。実データでは踏んでいない。
3. **Part A**（適応σ、密な側）は未着手。Part B より難しく、論文値も MAE 68.2 と大きい。
4. **FIDTM のハイパラ**（α=0.02, β=0.75, ピーク閾値 0.5, NMS 半径 1）は論文既定のまま。
   実測で詰めていない。
