# Papers

GNGおよび関連アルゴリズムの論文リストです。

## Growing Neural Gas (GNG)

### 基本GNG
- **Fritzke, B. (1995)**. "A Growing Neural Gas Network Learns Topologies"
  - NIPS 1994
  - [PDF](https://proceedings.neurips.cc/paper/1994/file/d56b9fc4b0f1be8571f8e2db7ae49e87-Paper.pdf)

### GNG-U (Utility)
- **Fritzke, B. (1997)**. "A Self-Organizing Network that Can Follow Non-Stationary Distributions"
  - ICANN 1997

### GNG-T (Triangulation)
- **Kubota, N. & Satomi, M. (2008)**. "自己増殖型ニューラルネットワークと教師無し分類学習"
  - 計測と制御, Vol.47, No.1
  - [PDF](papers/gng/kubota_2008_gng_t.pdf)
  - 備考: GNG-Tの三角形分割アルゴリズム（四角形探索・交差点探索）を提案

### AiS-GNG (Add-if-Silent)
- **Shoji, M., Obo, T., & Kubota, N. (2023)**. "Add-if-Silent Rule-Based Growing Neural Gas for High-Density Topological Structure of Unknown Objects"
  - IEEE RO-MAN 2023
  - DOI: 10.1109/RO-MAN57019.2023.10309556
  - [PDF](papers/gng/shoji_2023_ais_gng.pdf)
  - 備考: 遠方物体に高密度ノードを生成するためのAdd-if-Silentルール

### AiS-GNG-AM (Add-if-Silent with Amount of Movement)
- **Shoji, M., Obo, T., & Kubota, N. (2023)**. "Add-if-Silent Rule-Based Growing Neural Gas with Amount of Movement for High-Density Topological Structure Generation of Dynamic Object"
  - IEEE SMC 2023
  - DOI: 10.1109/SMC53992.2023.10394107
  - [PDF](papers/gng/shoji_2023_ais_gng_am.pdf)
  - 備考: 動的/静的物体の識別機能を追加

## SOINN Family

### SOINN
- **Furao, S., & Hasegawa, O. (2006)**. "An incremental network for on-line unsupervised classification and topology learning"
  - Neural Networks, 19(1), 90-106
  - DOI: 10.1016/j.neunet.2005.04.006

### Enhanced SOINN (E-SOINN)
- **Furao, S., Ogura, T., & Hasegawa, O. (2007)**. "An enhanced self-organizing incremental neural network for online unsupervised learning"
  - Neural Networks, 20(8), 893-903

## Other Related Algorithms

### Grow When Required (GWR)
- **Marsland, S., Shapiro, J., & Nehmzow, U. (2002)**. "A self-organising network that grows when required"
  - Neural Networks, 15(8-9), 1041-1058

### Neural Gas
- **Martinetz, T., & Schulten, K. (1991)**. "A 'Neural-Gas' Network Learns Topologies"
  - Artificial Neural Networks, 397-402

## 3D Point Cloud Applications

(関連論文を追加予定)

---

## フォルダ構成

```
references/
├── papers/
│   └── gng/                    # GNG系論文
│       ├── kubota_2008_gng_t.pdf
│       ├── shoji_2023_ais_gng.pdf
│       └── shoji_2023_ais_gng_am.pdf
├── notes/                      # アルゴリズムノート
├── original_code/              # リファレンス実装
└── papers.md                   # このファイル
```

## 論文の追加方法

```markdown
### タイトル
- **著者 (年)**. "論文タイトル"
  - 掲載誌/会議
  - DOI/URL:
  - [PDF](papers/[category]/filename.pdf)
  - 備考:
```
