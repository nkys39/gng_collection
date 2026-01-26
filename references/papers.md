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

### GNG-U2 (Utility V2)
- **Toda, Y., Ju, Z., Yu, H., Takesue, N., Wada, K., & Kubota, N. (2016)**. "Real-time 3D Point Cloud Segmentation using Growing Neural Gas with Utility"
  - IEEE SMC 2016
  - [PDF](papers/gng/toda_2016_gng_u2.pdf)
  - 備考: GNG-Uを3D点群セグメンテーションに適用、κ間隔でのUtilityチェック

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

### GNG Optimization
- **Fišer, D., Faigl, J., & Kulich, M. (2013)**. "Growing Neural Gas Efficiently"
  - Neurocomputing, Vol.104, pp.72-82
  - DOI: 10.1016/j.neucom.2012.10.004
  - [PDF](papers/gng/2013_Fiser_etal_Neurocomputing_GNG_Efficiently.pdf)
  - 備考: GNGの高速化技法（O-Tree、フラグベース近傍探索）

## 3D Surface Reconstruction

### GCS based
- **Ivrissimtzis, I.P., Jeong, W-K., & Seidel, H-P. (2003)**. "Using Growing Cell Structures for Surface Reconstruction"
  - Shape Modeling International (SMI) 2003
  - [PDF](papers/gng/2003_Ivrissimtzis_etal_SMI_GCS_Surface_Reconstruction.pdf)
  - 備考: GCSを点群からの表面再構成に適用、シャープエッジや凹部に対応

### Growing Self-Organizing Maps (GSRM)
- **Do Rêgo, R.L.M.E., Araújo, A.F.R., & De Lima Neto, F.B. (2007)**. "Growing Self-Reconstruction Maps"
  - IEEE IJCNN 2007
  - [PDF](papers/gng/2007_Rego_etal_IJCNN_GSRM_Surface_Reconstruction.pdf)
  - 備考: Growing SOMを用いた表面再構成、凹部や穴への対応

### GNG for 3D Reconstruction
- **Orts-Escolano, S., Garcia-Rodriguez, J., Morell, V., Cazorla, M., Serra Perez, J.A., & Garcia Garcia, A. (2016)**. "3D surface reconstruction of noisy point clouds using Growing Neural Gas"
  - Neural Processing Letters, Vol.43(2), pp.401-423
  - DOI: 10.1007/s11063-015-9421-x
  - [PDF](papers/gng/2016_Orts-Escolano_etal_NeuralProcessLett_3D_Surface_Reconstruction_GNG.pdf)
  - 備考: 低価格3Dセンサからのノイズ点群に対応、色・法線情報を学習に利用

### GNG for Robotics
- **Mueller, C.A., Hochgeschwender, N., & Ploeger, P.G. (2011)**. "Surface Reconstruction with Growing Neural Gas"
  - IROS 2011 Workshop on Active Semantic Perception
  - [PDF](papers/gng/Mueller_etal_Surface_Reconstruction_GNG_Robotics.pdf)
  - 備考: 家庭内サービスロボット向け、ノイズ耐性のある表面再構成

### sGNG vs GCS Comparison
- **AMC Bridge (2018)**. "Surface Reconstruction based on Neural Network"
  - Tech Report
  - [PDF](papers/gng/2018_AMCBridge_TechReport_sGNG_GCS_Surface_Reconstruction.pdf)
  - 備考: sGNG（Surface GNG）とGCSの比較、実装最適化手法

---

## フォルダ構成

```
references/
├── papers/
│   └── gng/                                                    # GNG系論文
│       ├── kubota_2008_gng_t.pdf                               # GNG-T (Kubota)
│       ├── toda_2016_gng_u2.pdf                                # GNG-U2 (Toda)
│       ├── shoji_2023_ais_gng.pdf                              # AiS-GNG
│       ├── shoji_2023_ais_gng_am.pdf                           # AiS-GNG-AM
│       ├── 2013_Fiser_etal_Neurocomputing_GNG_Efficiently.pdf  # GNG Optimization
│       ├── 2003_Ivrissimtzis_etal_SMI_GCS_Surface_Reconstruction.pdf
│       ├── 2007_Rego_etal_IJCNN_GSRM_Surface_Reconstruction.pdf
│       ├── 2016_Orts-Escolano_etal_NeuralProcessLett_3D_Surface_Reconstruction_GNG.pdf
│       ├── 2018_AMCBridge_TechReport_sGNG_GCS_Surface_Reconstruction.pdf
│       └── Mueller_etal_Surface_Reconstruction_GNG_Robotics.pdf
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
