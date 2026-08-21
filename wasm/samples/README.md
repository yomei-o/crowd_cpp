# wasm/samples

デモ用のサンプル写真。**ShanghaiTech の画像は入れていない**（データセットは各自で用意する方針、
かつ再配布の可否がはっきりしない）。代わりに Wikimedia Commons のパブリックドメイン / CC BY の
写真を、長辺 512 に縮小して置いてある。長辺 512 なのは:

* デモの既定の入力サイズと同じで、余計な縮小が入らない（頭の大きさが学習時に近いまま）
* 通常版モデルは 512x384 が wasm32 で通る上限（`RESUME.md` の「WASM デモ」に実測表）

縮小時に EXIF は落としてある（撮影地の位置情報を配らないため）。

| ファイル | 出典（Wikimedia Commons） | ライセンス | 作者 | この重みでの検出数 |
|---|---|---|---|---|
| `shibuya-crossing.jpg` | [Shibuya Station Crossing.JPG](https://commons.wikimedia.org/wiki/File:Shibuya_Station_Crossing.JPG) | **Public domain** | Picturetokyo (English Wikipedia) | 193 |
| `shibuya-scramble.jpg` | [Shibuya Crossing (52772885875).jpg](https://commons.wikimedia.org/wiki/File:Shibuya_Crossing_(52772885875).jpg) | **CC BY 2.0** | relux. | 87 |
| `shibuya-street.jpg` | [Shibuya Crossing (51784580942).jpg](https://commons.wikimedia.org/wiki/File:Shibuya_Crossing_(51784580942).jpg) | **CC BY 2.0** | Dick Thomas Johnson (Tokyo, Japan) | 42 |

検出数は `models/fidt_partB.onnx` に参考実装の LMDS を掛けた実測値（正解の人数ではない）。
目で見て点が人に乗っていることは確認済み。

**このモデルの得意・不得意がサンプルに出ている**: `shibuya-street.jpg` の手前に写っている
着ぐるみの人は検出されない。ShanghaiTech Part B は頭が 15〜30px の街頭写真なので、
カメラに近い大きな人物は学習分布の外にある。デモとしては「そういうもの」として見せている。
