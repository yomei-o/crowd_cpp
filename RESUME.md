# RESUME — crowd_cpp の進捗と残作業

方針は [README.md](README.md)。ここは**現在地・次の一手・未確認事項**を書き続けるファイル。

## 現在地（2026-08-20）

エンジンを姉妹リポから移植し、**足りなかった dilated conv を足して**、CSRNet のグラフを C++ が
書けるところまで。まだ学習していないので数字は「配管が通っている」ことの確認だけ。

| 検証 | 結果 |
|---|---|
| dilated conv の勾配（新規） | dilation 1/2/3 で解析勾配と中心差分が 3e-04 以下一致 |
| CSRNet のパラメータ数 | 16,263,489 = 論文の 16.26M |
| ONNX 妥当性 | `onnx.checker` PASS、onnxruntime 実行可 `[1,3,384,384]` → `[1,1,48,48]` |
| 自作ランタイム ⇔ onnxruntime | 同一入力で count 338.27 / 338.2708、max 0.1532 / 0.153233 |

## マイルストーン

- [x] **M0 足場とエンジン** — `pure/` を yolo_lpr_cpp から移植（autograd / nd / onnx / onnx_run /
      onnx_train / optim / ptio / trainrt / backend）。ビルドは g++ で通る。
- [x] **M1 dilated convolution** — `conv2d(..., dil)` と ONNX の `dilations` 属性。
      勾配は forward だけ見ても分からない（backward に 2 箇所入る）ので `pure/gradcheck.cpp` で縛る。
- [x] **M2 CSRNet のグラフ生成（C++）** — `crowd init-csrnet`。`--from-pt` で torchvision の
      VGG-16 から前段を転移できる形にした（パラメータ名を state_dict 名に合わせてある）。
- [ ] **M3 VGG-16 転移の検証** — `--from-pt vgg16.pth` で前段を載せ、torchvision の前段出力と
      突き合わせる（1e-4 以内）。ランダム初期化では ReLU が死んでマップが一定になるので、
      ここを通さないと学習が始まらない。
- [ ] **M4 データ** — ShanghaiTech Part A/B。点アノテーションが `.mat`（MATLAB v5）なので、
      両言語で読める形にするコンバータが要る。密度ラベル（ガウシアン）生成も両言語。
- [ ] **M5 学習（両言語）** — MSE で密度回帰。Python は torch、C++ は ONNX グラフ直接学習
      （姉妹リポと同じ方式）。パリティは step1 の loss。
- [ ] **M6 評価** — 数え上げ MAE/MSE。ShanghaiTech A の CSRNet 論文値は MAE 68.2、B は 10.6。
- [ ] **M7 FIDTM** — ラベル生成（焦点逆距離変換）と後処理（局所最大）を差し替え、位置の F1 を測る。
- [ ] **M8 軽量版** — `--width` で細くしたものを学習し、WASM デモに載せる。
- [ ] **M9 P2PNet** — M7 の F1 が足りなければ。Hungarian マッチングを両言語に。

## 次の一手

1. **VGG-16 の重みを取ってきて `--from-pt` を通す**（M3）。torchvision の `vgg16.pth` は
   `features.<n>.weight` という名前で、こちらのグラフもその名前で作ってあるので載るはず。
   検証は「前段の出力を torchvision と比べて 1e-4」。
2. ShanghaiTech の取得と `.mat` リーダ（M4）。Kaggle にミラーがあるので、姉妹リポの kbridge 経路が使える。
3. 密度ラベル生成を両言語で（ガウシアン σ は固定 15 か適応 kNN。CSRNet 論文は Part A が適応、B が固定 15）。

## 未確認事項

1. **入力サイズ**: CSRNet は本来可変サイズ（畳み込みだけなので）だが、こちらの ONNX は静的形状で書いている。
   学習はクロップで固定サイズにできるが、評価は画像丸ごと入れたい。**静的形状のまま複数サイズの
   グラフを書き出す**か、ランタイムを可変形状にするかは未決。
2. **メモリ**: 384x384 で 16.3M パラメータ、活性値も大きい。CPU の自作エンジンで batch 何枚まで
   回るか未測定。
3. **適応ガウシアンの kNN**: Part A の密度ラベルは各点の最近傍 k 個の平均距離で σ を決める。
   両言語で同じ結果にするには近傍探索の順序まで決める必要がある（同距離の扱い）。

## ログ

- **2026-08-20** — プロジェクト開始。手法比較の上で CSRNet → FIDTM → （必要なら）P2PNet の順に決定。
  エンジン移植、dilated conv 追加（勾配検証つき）、CSRNet グラフ生成と ONNX 検証まで。
