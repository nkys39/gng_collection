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

---

## Batch Learning GNG

### BL-GNG (Batch Learning GNG)
- **Toda, Y., Chin, W., & Kubota, N. (2017)**. "Unsupervised Neural Network based Topological Learning from Point Clouds for Map Building"
  - Conference paper
  - [PDF](papers/gng/Toda et al. - 2017 - Unsupervised neural network based topological learning from point clouds for map building.pdf)
  - 備考: Fuzzy C-meansに基づく目的関数でBL-GNGを提案、学習収束を改善

### MBL-GNG (Multilayer Batch Learning GNG)
- **Toda, Y., Matsuno, T., & Minami, M. (2021)**. "Multilayer Batch Learning Growing Neural Gas for Learning Multiscale Topologies"
  - JACIII (Journal of Advanced Computational Intelligence and Intelligent Informatics)
  - [PDF](papers/gng/Toda et al. - 2021 - Multilayer Batch Learning Growing Neural Gas for Learning Multiscale Topologies.pdf)
  - 備考: 多層構造でマルチスケールトポロジーを学習

### MS-BL-GNG (Multi-Scale Batch Learning GNG)
- **Authors (2021)**. "Multi-scale Batch-Learning Growing Neural Gas for Topological Feature Extraction"
  - IWACIII 2021 (Beijing)
  - [PDF](papers/gng/2021 - Multi-Scale Batch-Learning Growing Neural Gas for Topological Feature Extraction in Navigation of Mo.pdf)
  - 備考: ナビゲーション向けマルチスケール特徴抽出

### Sequential BL-GNG
- **Ardilla, F., Saputra, A.A., & Kubota, N. (2022)**. "Batch Learning Growing Neural Gas for Sequential Point Cloud Processing"
  - Conference paper
  - [PDF](papers/gng/Ardilla et al. - 2022 - Batch Learning Growing Neural Gas for Sequential Point Cloud Processing.pdf)
  - 備考: 動的に変化する点群のトポロジー保存マップ構築

### MS-BL-GNG Efficiently
- **Ardilla, F., Saputra, A.A., & Kubota, N. (2023)**. "Multi-Scale Batch-Learning Growing Neural Gas Efficiently for Dynamic Data Distributions"
  - IJAT (International Journal of Automation Technology)
  - [PDF](papers/gng/Ardilla et al. - 2023 - Multi-Scale Batch-Learning Growing Neural Gas Efficiently for Dynamic Data Distributions.pdf)
  - 備考: 動的データ分布に対する効率的なMS-BL-GNG

### Distributed BL-GNG
- **Siow, C.Z., Saputra, A.A., Obo, T., & Kubota, N. (2024)**. "Distributed Batch Learning of Growing Neural Gas for Quick and Efficient Clustering"
  - Mathematics 2024, 12, 1909
  - DOI: 10.3390/math12121909
  - [PDF](papers/gng/Siow et al. - 2024 - Distributed Batch Learning of Growing Neural Gas for Quick and Efficient Clustering.pdf)
  - 備考: 分散バッチ学習による高速クラスタリング

---

## ROI-GNG (Region of Interest)

### ROI-GNG
- **Toda, Y., Li, X., Matsuno, T., & Minami, M. (2019)**. "Region of Interest Growing Neural Gas for Real-time Point Cloud Processing"
  - Conference paper
  - [PDF](papers/gng/Toda et al. - 2019 - Region of Interest Growing Neural Gas for Real-Time Point Cloud Processing.pdf)
  - 備考: 集中/分散センシングによるリアルタイム点群処理

### ROI-GNG Study (Japanese)
- **Toda, Y., Matsuno, T., & Minami, M.**. "Study of Region of Interest Growing Neural Gas"
  - Japanese paper
  - [PDF](papers/gng/Toda_etal_ROI-GNG_Study.pdf)
  - 備考: ROI-GNGの有効性検証

---

## GNG-DT (Different Topologies) & ART-based

### GNG-DT
- **Toda, Y., Wada, A., Miayase, H., Ozasa, K., Matsuno, T., & Minami, M. (2022)**. "Growing Neural Gas with Different Topologies for 3D Space Perception"
  - Applied Sciences 2022, 12, 1705
  - DOI: 10.3390/app12031705
  - [PDF](papers/gng/Toda et al. - 2022 - Growing Neural Gas with Different Topologies for 3D Space Perception.pdf)
  - 備考: 異なるトポロジー構造を持つGNGによる3D空間知覚

### ART-based Global Topological Map
- **Toda, Y. & Masuyama, N. (2024)**. "Adaptive Resonance Theory-Based Global Topological Map Building for an Autonomous Mobile Robot"
  - IEEE Access
  - DOI: 10.1109/ACCESS.2024.3442304
  - [PDF](papers/gng/Toda_Masuyama_2024_IEEE_Access_ART_Global_Topological_Map.pdf)
  - 備考: 破滅的忘却を回避するATC-DTによるグローバルトポロジカルマップ構築

### MLATC (Multilayer ATC)
- **Ofuchi, R., Toda, Y., Masuyama, N., & Matsuno, T. (2025)**. "MLATC: Fast Hierarchical Topological Mapping from 3D LiDAR Point Clouds Based on Adaptive Resonance Theory"
  - Journal paper (submitted/accepted)
  - [PDF](papers/gng/Ofuchi et al. - 2025 - MLATC Fast Hierarchical Topological Mapping from 3D LiDAR Point Clouds Based on Adaptive Resonance.pdf)
  - 備考: 大規模・動的環境向けの高速階層トポロジカルマッピング

---

## Robotics Applications

### 3D Maps / SLAM
- **Moreli, V., et al. (2014)**. "3D maps representation using GNG"
  - Conference paper
  - [PDF](papers/gng/Moreli et al. - 2014 - 3D maps representation using GNG.pdf)
  - 備考: GNGによる3Dマップ表現

- **Viejo, D., Garcia-Rodriguez, J., & Cazorla, M. (2014)**. "Combining visual features and Growing Neural Gas networks for robotic 3D SLAM"
  - Information Sciences
  - [PDF](papers/gng/Viejo_etal_2014_InfSci_GNG_3D_SLAM.pdf)
  - 備考: GNGとビジュアル特徴を組み合わせた3D SLAM

### Point Cloud Sequences
- **Orts-Escolano, S., Garcia-Rodriguez, J., Morell, V., Cazorla, M., Saval, M., & Azorin, J. (2015)**. "Processing Point Cloud Sequences with Growing Neural Gas"
  - Conference paper
  - [PDF](papers/gng/Orts-Escolano et al. - 2015 - Processing point cloud sequences with Growing Neural Gas.pdf)
  - 備考: Kinectによる動的シーンの点群シーケンス処理、物体追跡

### Fast-GNG for Mobile Robots
- **Junaedy, et al. (2022)**. "Object Extraction Method for Mobile Robots using Fast Growing Neural Gas"
  - Conference paper
  - [PDF](papers/gng/Junaedy et al. - 2022 - Object Extraction Method for Mobile Robots using Fast Growing Neural Gas.pdf)
  - 備考: モバイルロボット向け高速GNGによる物体抽出

### Event-based Camera
- **Doteguchi & Kubota (2022)**. "Topological Mapping for Event-based camera using Fast-GNG and SNN"
  - Conference paper
  - [PDF](papers/gng/Doteguchi と Kubota - 2022 - Topological Mapping for Event-based camera using Fast-GNG and SNN.pdf)
  - 備考: イベントカメラとSNNを組み合わせたFast-GNGトポロジカルマッピング

### Traversability Clustering
- **Ozasa, K., Toda, Y., & Matsuno, T. (2023)**. "Growing Neural Gas based Traversability Clustering for an Autonomous Robot"
  - Conference paper
  - [PDF](papers/gng/Ozasa et al. - 2023 - Growing Neural Gas based Traversability Clustering for an Autonomous Robot.pdf)
  - 備考: 自律ロボット向け走行可能領域クラスタリング

---

## Quadruped Robot / Ladder Detection

### DD-GNG (Dynamic Density GNG)
- **Saputra, A.A., et al. (2019)**. "Dynamic Density Topological Structure Generation for Real-Time Ladder Affordance Detection"
  - Conference paper
  - [PDF](papers/gng/Saputra et al. - 2019 - Dynamic Density Topological Structure Generation for Real-Time Ladder Affordance Detection.pdf)
  - 備考: 4脚ロボット向けリアルタイム梯子検出

### Multi-Level Control
- **Saputra, A.A., et al. (2022)**. "Topological based Environmental Reconstruction for Efficient Multi-Level Control of Robot Locomotion"
  - Conference paper
  - [PDF](papers/gng/Saputra et al. - 2022 - Topological based Environmental Reconstruction for Efficient Multi-Level Control of Robot Locomotion.pdf)
  - 備考: トポロジカル環境再構成によるロボット歩行制御

### Foothold Planning
- **Saputra, A.A., Shoji, M., Ardilla, F., & Kubota, N. (2024)**. "Fast Foothold Planning for Quadruped Robot Locomotion Using Topological Perception"
  - Conference paper
  - [PDF](papers/gng/Saputra et al. - 2024 - Fast Foothold Planning for Quadruped Robot Locomotion Using Topological Perception.pdf)
  - 備考: 4脚ロボット向け高速足場計画

### Fuzzy Reliability-Based Growing Region
- **Shoji, M., Watanabe, T., Nagashima, K., Michikawa, R., & Kubota, N. (2024)**. "Topological Clustering for Spatial Perception Using Fuzzy Reliability-Based Growing Region Method"
  - Conference paper
  - [PDF](papers/gng/Shoji et al. - 2024 - Topological Clustering for Spatial Perception Using Fuzzy Reliability-Based Growing Region Method.pdf)
  - 備考: ファジー信頼性に基づく空間知覚クラスタリング

---

## 3D Shape Recognition

### Contour Detection
- **Furuta, Y., Toda, Y., & Matsuno, T.**. "Study of Contour Detection for Growing Neural Gas based 3D Shape Recognition"
  - Japanese paper (Okayama University)
  - [PDF](papers/gng/Furuta_etal_GNG_Contour_Detection_3D_Shape_Recognition.pdf)
  - 備考: GNGの位相構造からの輪郭検出、物体把持点検出向け

---

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

### GNG for Service Robotics
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

## GNG Optimization

- **Fišer, D., Faigl, J., & Kulich, M. (2013)**. "Growing Neural Gas Efficiently"
  - Neurocomputing, Vol.104, pp.72-82
  - DOI: 10.1016/j.neucom.2012.10.004
  - [PDF](papers/gng/2013_Fiser_etal_Neurocomputing_GNG_Efficiently.pdf)
  - 備考: GNGの高速化技法（O-Tree、フラグベース近傍探索）

---

## SOINN Family

### SOINN
- **Furao, S., & Hasegawa, O. (2006)**. "An incremental network for on-line unsupervised classification and topology learning"
  - Neural Networks, 19(1), 90-106
  - DOI: 10.1016/j.neunet.2005.04.006

### Enhanced SOINN (E-SOINN)
- **Furao, S., Ogura, T., & Hasegawa, O. (2007)**. "An enhanced self-organizing incremental neural network for online unsupervised learning"
  - Neural Networks, 20(8), 893-903

---

## Other Related Algorithms

### Grow When Required (GWR)
- **Marsland, S., Shapiro, J., & Nehmzow, U. (2002)**. "A self-organising network that grows when required"
  - Neural Networks, 15(8-9), 1041-1058

### Neural Gas
- **Martinetz, T., & Schulten, K. (1991)**. "A 'Neural-Gas' Network Learns Topologies"
  - Artificial Neural Networks, 397-402

---

## フォルダ構成

```
references/papers/gng/
├── kubota_2008_gng_t.pdf
├── toda_2016_gng_u2.pdf
├── shoji_2023_ais_gng.pdf
├── shoji_2023_ais_gng_am.pdf
├── 2013_Fiser_etal_Neurocomputing_GNG_Efficiently.pdf
├── 2003_Ivrissimtzis_etal_SMI_GCS_Surface_Reconstruction.pdf
├── 2007_Rego_etal_IJCNN_GSRM_Surface_Reconstruction.pdf
├── 2016_Orts-Escolano_etal_NeuralProcessLett_3D_Surface_Reconstruction_GNG.pdf
├── 2018_AMCBridge_TechReport_sGNG_GCS_Surface_Reconstruction.pdf
├── Mueller_etal_Surface_Reconstruction_GNG_Robotics.pdf
├── 2021 - Multi-Scale Batch-Learning Growing Neural Gas...pdf
├── Ardilla et al. - 2022 - Batch Learning Growing Neural Gas...pdf
├── Ardilla et al. - 2023 - Multi-Scale Batch-Learning Growing Neural Gas...pdf
├── Doteguchi と Kubota - 2022 - Topological Mapping...pdf
├── Furuta_etal_GNG_Contour_Detection_3D_Shape_Recognition.pdf
├── Junaedy et al. - 2022 - Object Extraction Method...pdf
├── Moreli et al. - 2014 - 3D maps representation using GNG.pdf
├── Ofuchi et al. - 2025 - MLATC Fast Hierarchical...pdf
├── Orts-Escolano et al. - 2015 - Processing point cloud sequences...pdf
├── Ozasa et al. - 2023 - Growing Neural Gas based Traversability...pdf
├── Saputra et al. - 2019 - Dynamic Density Topological Structure...pdf
├── Saputra et al. - 2022 - Topological based Environmental...pdf
├── Saputra et al. - 2024 - Fast Foothold Planning...pdf
├── Shoji et al. - 2024 - Topological Clustering...pdf
├── Siow et al. - 2024 - Distributed Batch Learning...pdf
├── Toda et al. - 2017 - Unsupervised neural network...pdf
├── Toda et al. - 2019 - Region of Interest Growing Neural Gas...pdf
├── Toda et al. - 2021 - Multilayer Batch Learning...pdf
├── Toda et al. - 2022 - Growing Neural Gas with Different Topologies...pdf
├── Toda_Masuyama_2024_IEEE_Access_ART_Global_Topological_Map.pdf
├── Toda_etal_ROI-GNG_Study.pdf
└── Viejo_etal_2014_InfSci_GNG_3D_SLAM.pdf
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
