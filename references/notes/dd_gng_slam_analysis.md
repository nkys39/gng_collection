# DD-GNG SLAM機能解析

リファレンス実装 `azhar_ddgng` および `toda_gngdt` に含まれるSLAM関連機能の詳細解析。

## 概要

DD-GNGリファレンス実装には、GNGノードの法線ベクトルを活用した**GNGベースSLAM**が含まれています。従来のLiDAR点群SLAMとは異なり、GNGで圧縮したトポロジカル表現を使用します。

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                    GNG-based SLAM System                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────────┐  │
│  │   Current    │────▶│   Surface    │────▶│   Global    │  │
│  │   GNG Frame  │     │   Matching   │     │    Map      │  │
│  │  mapData[1]  │     │              │     │  mapData[0] │  │
│  └──────────────┘     └──────────────┘     └─────────────┘  │
│         │                    │                     ▲        │
│         │                    ▼                     │        │
│         │             ┌──────────────┐             │        │
│         │             │  Transform   │             │        │
│         └────────────▶│  Estimation  │─────────────┘        │
│                       │ SLAMpos/rot  │                      │
│                       └──────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

## データ構造

### GNG構造体 (`gng.hpp`)

```c
struct gng {
    // 基本GNGデータ
    double node[GNGN][DIM];           // ノード位置 (x, y, z)
    int edge[GNGN][GNGN];             // エッジ接続行列
    int age[GNGN][GNGN];              // エッジ年齢
    double gng_err[GNGN];             // 積算誤差
    double gng_u[GNGN];               // Utility値
    double strength[GNGN];            // ノード強度

    // 法線・サーフェス分類
    double normVect[GNGN][5];         // 法線ベクトル [nx, ny, nz, ?, ?]
    double normTriangle[GNGN][DIM];   // サーフェス分類 [type, avg_edge_length, ?]
    int normAge[GNGN];                // サーフェス分類の安定性カウンタ

    // SLAM用マップデータ
    double mapData[2][30*GNGN][9];    // マップノードデータ
    int mapEdge[2][30*GNGN][30*GNGN]; // マップエッジ接続
    int n_mapData[2];                 // マップノード数
    int mapAge[GNGN];                 // マップノードの観測年齢

    // その他
    int susNode[GNGN];                // 注目ノード（コーナー等）リスト
    int susNode_n;                    // 注目ノード数
    int susNodeLabel[GNGN];           // 注目ノードのクラスタラベル
    // ...
};
```

### mapData配列の構造

```
mapData[frame_index][node_index][attribute_index]

frame_index:
  0 = グローバルマップ（累積）
  1 = 現在フレームのGNGデータ

attribute_index:
  0-2: 位置 (x, y, z) - ワールド座標系
  3-5: 法線ベクトル (nx, ny, nz) - ワールド座標系
  6:   サーフェスタイプ (0-6)
  7:   強度 × 10 (normTriangle[i][1] * 10)
  8:   未使用
```

### サーフェスタイプ値

| 値 | 名称 | 条件 | 説明 |
|----|------|------|------|
| 0 | PLANE | 法線長 ≥ 0.047 かつ 角度 ≤ π/12 | 平坦な平面領域 |
| 1 | EDGE | 法線長 ≥ 0.047 かつ 角度 > π/12 | 傾斜や境界領域 |
| 2 | CORNER | 法線長 < 0.047 | 法線が収束しないコーナー |
| 3 | UNKNOWN | 接続数 ≤ 2 | 分類不可 |
| 4 | STABLE_PLANE | type=0 かつ normAge > 8 | 安定した平面 |
| 5 | STABLE_EDGE | type=1 かつ normAge > 16 | 安定したエッジ |
| 6 | STABLE_CORNER | type=2 かつ normAge > 16 | 安定したコーナー → **注目領域** |

## グローバル変数

```c
// SLAM状態 (surfMatching.hpp)
extern double SLAMpos[3];    // 累積位置 [x, y, z]
extern double SLAMrot[3];    // 累積回転 [roll, pitch, yaw] (radians)
extern double dSLAMrot[3];   // 回転差分
extern int stepMatch;        // Surface Matching処理ステップ (0-5)
```

## Surface Matching SLAM処理フロー

### メイン関数: `surfaceMatching(struct gng *net)`

```c
void surfaceMatching(struct gng *net) {
    switch(stepMatch) {
        case 0: // 初期化
            nD = 0; nE = 0;
            stepMatch = 1;
            break;

        case 1: // 特徴ノード選択
            generateAssignedNode(net);  // コーナー/エッジノードを選択
            stepMatch = 2;
            break;

        case 2: // 対応点探索
            findNearestOriginNode(net); // マップとのマッチング
            stepMatch = 3;
            break;

        case 3: // 変換推定（反復）
            while(maxError > 0.003 && count < 200) {
                maxError = vectorMatching(net);
                count++;
            }
            stepMatch = 4;
            break;

        case 4: // マップ更新
            add_n_changeOriginNode(net); // 新ノードをマップに追加
            calcAgeMapNode(net);         // ノード年齢更新
            stepMatch = 5;
            break;
    }
}
```

### Step 1: 特徴ノード選択 (`generateAssignedNode`)

```c
void generateAssignedNode(struct gng *net) {
    for (int i = 0; i < net->n_mapData[1]; i += a) {
        a = 1 + (int)(1 * rnd());  // ランダムスキップ

        // サーフェスタイプ 1, 4, 5, 7 のノードを選択
        // (EDGE, STABLE_PLANE, STABLE_EDGE, or custom type)
        if (net->mapData[1][i][6] == 1 ||
            net->mapData[1][i][6] == 4 ||
            net->mapData[1][i][6] == 5 ||
            net->mapData[1][i][6] == 7) {
            D[nD] = i;  // 選択されたノードのインデックス
            nD++;
        }
    }
}
```

**特徴**:
- コーナーやエッジなど**幾何的特徴が明確なノード**のみを使用
- 平面ノードは対応点探索で曖昧性が高いため除外
- ランダムサンプリングで計算量削減

### Step 2: 対応点探索 (`findNearestOriginNode`)

```c
void findNearestOriginNode(struct gng *net) {
    for (int i = 0; i < nD; i++) {
        // 現在フレームのノード位置・法線を取得
        for (int k = 0; k < 3; k++) {
            n1[k] = net->mapData[1][D[i]][k];      // 位置
            v1[k] = net->mapData[1][D[i]][k+3];    // 法線
        }

        E[i] = -1;  // 対応なし
        double min = 1000;

        for (int j = 0; j < net->n_mapData[0]; j++) {
            // サーフェスタイプ >= 4 のマップノードのみ
            if (net->mapData[0][j][6] >= 4) {
                // 位置の距離計算
                double a = norm(vectorSubtraction(n1, n2, 3), 3);

                if (a < min && a < 0.3) {  // 距離閾値
                    // 法線の向きチェック（逆向きであるべき）
                    double b = norm(vectorAdd(v1, v2, 3), 3);

                    if (b > 0.088) {  // 法線が逆向き
                        min = a;
                        E[i] = j;
                        net->mapAge[j] = 0;  // 観測されたのでリセット
                    }
                }
            }
        }
    }
}
```

**マッチング条件**:
1. **距離**: ||p_current - p_map|| < 0.3
2. **法線**: ||n_current + n_map|| > 0.088（逆向きのサーフェス）
3. **サーフェスタイプ**: マップ側は安定タイプ（≥4）のみ

### Step 3: 変換推定 (`vectorMatching`)

法線ベクトルの差から回転を、位置の差から並進を推定。

```c
double vectorMatching(struct gng *net) {
    double EuAngle[3] = {0, 0, 0};        // オイラー角変化
    double deltaNodeMove[3] = {0, 0, 0};  // 並進変化
    double maxEffect = 0;

    // 重みの合計を計算
    for (int i = 0; i < nD; i++) {
        if (E[i] >= 0) {
            maxEffect += net->mapData[1][D[i]][7];  // 強度で重み付け
        }
    }

    for (int i = 0; i < nD; i++) {
        if (E[i] >= 0) {
            // 位置と法線を取得
            for (int k = 0; k < 3; k++) {
                n1[k] = net->mapData[1][D[i]][k];      // 現在位置
                v1[k] = net->mapData[1][D[i]][k+3];    // 現在法線
                n2[k] = net->mapData[0][E[i]][k];      // マップ位置
                v2[k] = net->mapData[0][E[i]][k+3];    // マップ法線
            }

            // ===== 並進推定 =====
            // 位置差を法線方向に射影
            double *a = vectorSubtraction(n2, n1, 3);  // Δp = p_map - p_current
            double *n = vectorUnit(v1);                // 法線単位ベクトル
            double a1 = dotProduct(a, n);              // 法線方向成分
            double *A1 = vectorScale(a1, n, 3);        // 射影ベクトル

            // 重み付き並進を累積
            for (int k = 0; k < 3; k++) {
                deltaNodeMove[k] += A1[k] * net->mapData[1][D[i]][7];
            }

            // ===== 回転推定 =====
            // 法線ベクトル間の回転をオイラー角に変換
            double ang[3];
            vectorToEuler(v1, v2, ang);

            // 重み付き回転を累積
            for (int k = 0; k < 3; k++) {
                EuAngle[k] += ang[k] * net->mapData[1][D[i]][7];
            }
        }
    }

    // ===== 変換の適用 =====
    if (cE >= 1) {
        // 回転行列を計算
        for (int k = 0; k < 3; k++) {
            if (EuAngle[k] > 0.001*cE || EuAngle[k] < -0.001*cE)
                a[k] = 0.3 * EuAngle[k] / maxEffect;  // 減衰係数0.3
            else
                a[k] = 0;
        }
        EulerToMatrix(a[0], a[1], a[2], F);

        // 現在フレームの全ノードを回転
        for (int i = 0; i < net->n_mapData[1]; i++) {
            // 位置を回転
            vectorFromMatrixRotation(F, v1, v2);
            // 法線も回転
            // ...
        }

        // 並進を適用
        for (int i = 0; i < net->n_mapData[1]; i++) {
            for (int k = 0; k < 3; k++) {
                net->mapData[1][i][k] += 0.3 * deltaNodeMove[k] / maxEffect;
            }
        }

        // グローバルSLAM状態を更新
        for (int k = 0; k < 3; k++) {
            SLAMpos[k] += 0.3 * deltaNodeMove[k] / maxEffect;
            SLAMrot[k] += a[k];
        }
    }

    return maxError / cE;
}
```

**変換推定のポイント**:
1. **Point-to-Plane距離**: 位置差を法線方向に射影（並進推定）
2. **法線マッチング**: Rodrigues回転でオイラー角を計算（回転推定）
3. **重み付け**: ノード強度（`mapData[i][7]`）で各マッチングを重み付け
4. **減衰係数**: 0.3 で急激な変化を抑制
5. **反復**: error < 0.003 または 200回まで反復

### Step 4: マップ更新 (`add_n_changeOriginNode`)

```c
void add_n_changeOriginNode(struct gng *net) {
    int n_map0 = net->n_mapData[0];
    int count = n_map0;

    for (int i = 0; i < net->n_mapData[1]; i++) {
        // 最近傍マップノードを探索
        // ...

        // 新規ノード追加条件:
        // - サーフェスタイプ >= 4 かつ < 6 (STABLE_PLANE or STABLE_EDGE)
        // - 最近傍距離 > 0.2
        if ((net->mapData[1][i][6] >= 4 &&
             net->mapData[1][i][6] < 6 &&
             near > 0.2)) {

            add[i] = 1;  // 追加フラグ

            // 近傍マップノードとエッジ接続
            for (int j = 0; j < ncon; j++) {
                net->mapEdge[0][con[j]][count] = 1;
                net->mapEdge[0][count][con[j]] = 1;
            }
            count++;
        }
    }

    // 追加ノードのデータをコピー
    count = n_map0;
    for (int i = 0; i < net->n_mapData[1]; i++) {
        if (add[i] == 1) {
            for (int k = 0; k < 8; k++) {
                net->mapData[0][count][k] = net->mapData[1][i][k];
            }
            // 現在フレーム内のエッジもコピー
            // ...
            count++;
        }
    }

    net->n_mapData[0] = count;
    net->n_mapData[1] = 0;  // 現在フレームをクリア
}
```

### GNGデータのマップ変換 (`getGNGdata`)

```c
void getGNGdata(struct gng *net, int c) {
    int count = 0;
    double F[3][3];
    double v1[3], v2[3];

    // 累積回転行列を計算
    EulerToMatrix(SLAMrot[0], SLAMrot[1], SLAMrot[2], F);

    for (int j = 0; j < net->node_n; j++) {
        // 未分類ノードはスキップ
        if (net->normTriangle[j][0] != 3) {
            // 位置をワールド座標系に変換
            for (int k = 0; k < 3; k++) {
                v1[k] = net->node[j][k];
            }
            vectorFromMatrixRotation(F, v1, v2);
            for (int k = 0; k < 3; k++) {
                net->mapData[c][count][k] = v2[k] + SLAMpos[k];
            }

            // 法線もワールド座標系に変換
            for (int k = 0; k < 3; k++) {
                v1[k] = net->normVect[j][k];
            }
            vectorFromMatrixRotation(F, v1, v2);
            for (int k = 3; k < 6; k++) {
                net->mapData[c][count][k] = v2[k-3];
            }

            // サーフェスタイプと強度
            net->mapData[c][count][6] = net->normTriangle[j][0];
            net->mapData[c][count][7] = 10 * net->normTriangle[j][1];

            // エッジ接続もコピー
            // ...

            count++;
        }
    }
    net->n_mapData[c] = count;
}
```

## ICP (Iterative Closest Point) 実装

`toda_gngdt/gng_livox/src/icp.c` に含まれるICP実装。GNG-DTで使用。

### アルゴリズム: Horn (1987) Unit Quaternion法

```
Input:  xyz1[cct[0]][3] - ソース点群
        xyz2[cct[1]][3] - ターゲット点群
Output: GQ[4] - 累積クォータニオン
        GT[3] - 累積並進

1. 対応点探索
2. 重心計算
3. 相関行列 → 4x4対称行列Q
4. Qの最大固有値の固有ベクトル → 回転クォータニオン
5. 並進 = μ_target - R * μ_source
6. 点群を変換して繰り返し
```

### 主要関数

#### 対応点探索

```c
int search_matchingPoint(double **xyz1, double **xyz2,
                         int mindex[], int cct[], int *checkflag) {
    for (int i = 0; i < cct[0]; i++) {
        int min_no = 0;
        double mindis = distance(xyz1[i], xyz2[0]);

        // 最近傍点を探索
        for (int j = 1; j < cct[1]; j++) {
            double dis = distance(xyz1[i], xyz2[j]);
            if (dis < mindis) {
                mindis = dis;
                min_no = j;
            }
        }

        // 距離閾値チェック
        if (mindis > 0.2 * 0.2) {
            checkflag[i] = 1;  // 外れ値
        } else {
            checkflag[i] = 0;  // 有効
        }
        mindex[i] = min_no;
    }
}
```

#### 重心計算

```c
int calc_medianpoint(int numberOfPairs,
                     double position[][3], double pairPosition[][3],
                     double *mup, double *muy,
                     int *checkflag, int sflag) {
    int ct = 0;
    for (int i = 0; i < 3; i++) {
        mup[i] = 0.0;
        muy[i] = 0.0;
    }

    for (int i = 0; i < numberOfPairs; i++) {
        if (checkflag[i] != 1) {  // 有効な対応のみ
            mup[0] += position[i][0];
            mup[1] += position[i][1];
            mup[2] += position[i][2];
            muy[0] += pairPosition[i][0];
            muy[1] += pairPosition[i][1];
            muy[2] += pairPosition[i][2];
            ct++;
        }
    }

    for (int i = 0; i < 3; i++) {
        mup[i] /= (double)ct;
        muy[i] /= (double)ct;
    }
    return ct;
}
```

#### 固有値分解による回転推定

```c
void calc_eigenvalue(int numberOfPairs,
                     double position[][3], double pairPosition[][3],
                     double *mup, double *muy,
                     double *q, int *checkflag) {
    double SIGP[3][3], Q[4][4];

    // 相関行列 SIGP = Σ (p_i - μ_p)(y_i - μ_y)^T
    memset(SIGP, 0, sizeof(SIGP));
    for (int i = 0; i < numberOfPairs; i++) {
        if (checkflag[i] == 0) {
            SIGP[0][0] += (position[i][0]-mup[0]) * (pairPosition[i][0]-muy[0]);
            SIGP[0][1] += (position[i][0]-mup[0]) * (pairPosition[i][1]-muy[1]);
            // ... 他の要素
        }
    }

    // 4x4対称行列Q (Horn 1987, Eq.26)
    Q[0][0] = SIGP[0][0] + SIGP[1][1] + SIGP[2][2];
    Q[0][1] = SIGP[1][2] - SIGP[2][1];
    Q[0][2] = SIGP[2][0] - SIGP[0][2];
    Q[0][3] = SIGP[0][1] - SIGP[1][0];
    // ... 対称要素

    // LAPACK dgeev_ で固有値分解
    dgeev_(&jobvl, &jobvr, &a, Q2, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, &info, 1, 1);

    // 最大固有値の固有ベクトル = 回転クォータニオン
    int maxn = 0;
    double max = wr[0];
    for (int i = 1; i < 4; i++) {
        if (max < wr[i]) {
            max = wr[i];
            maxn = i;
        }
    }

    for (int i = 0; i < 4; i++) {
        q[i] = vr[i + maxn * 4];
    }
}
```

#### 変換行列の計算

```c
void calc_transformation(double R[3][3], double *q2,
                         double *mup, double *muy, double *q) {
    // クォータニオン → 回転行列
    // R = (q0^2 + q1^2 - q2^2 - q3^2)  2(q1q2 - q0q3)           2(q1q3 + q0q2)
    //     2(q1q2 + q0q3)               (q0^2 - q1^2 + q2^2 - q3^2)  2(q2q3 - q0q1)
    //     2(q1q3 - q0q2)               2(q2q3 + q0q1)           (q0^2 - q1^2 - q2^2 + q3^2)

    double a[3], b[3], c[3];
    a[0] = q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3];
    // ... 他の要素

    // 直交化 (Gram-Schmidt)
    // ...

    R[0][0] = a[0]; R[0][1] = b[0]; R[0][2] = c[0];
    R[1][0] = a[1]; R[1][1] = b[1]; R[1][2] = c[1];
    R[2][0] = a[2]; R[2][1] = b[2]; R[2][2] = c[2];

    // 並進 = μ_target - R * μ_source
    q2[0] = muy[0] - R[0][0]*mup[0] - R[0][1]*mup[1] - R[0][2]*mup[2];
    q2[1] = muy[1] - R[1][0]*mup[0] - R[1][1]*mup[1] - R[1][2]*mup[2];
    q2[2] = muy[2] - R[2][0]*mup[0] - R[2][1]*mup[1] - R[2][2]*mup[2];
}
```

## 従来SLAMとの比較

| 特徴 | 従来の点群SLAM | GNGベースSLAM |
|------|---------------|---------------|
| データ表現 | 生点群 | GNGノード（圧縮） |
| データ量 | 数万〜数十万点 | 数百ノード |
| 特徴抽出 | 別処理（FPFH等） | 法線/サーフェス分類で統合 |
| マッチング | 点 ↔ 点 | ノード ↔ ノード（法線考慮） |
| 回転推定 | ICP | Surface Matching（法線ベース） |
| マップ管理 | KD-tree等 | GNGエッジ構造 |
| 計算量 | O(n log n) | O(N²) (N=ノード数<<n) |

## 強度 (Strength) の役割

SLAM処理における強度の利用：

1. **マッチング重み付け**: 高強度ノードのマッチングを優先
2. **変換推定**: 高強度ノードの寄与を増大
3. **マップ更新**: 強度情報も伝搬

```c
// vectorMatching内
for (int k = 0; k < 3; k++) {
    deltaNodeMove[k] += A1[k] * net->mapData[1][D[i]][7];  // 強度で重み付け
    EuAngle[k] += ang[k] * net->mapData[1][D[i]][7];
}
```

## パラメータ

### Surface Matching

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| 距離閾値 | 0.3 | マッチング最大距離 |
| 法線閾値 | 0.088 | 逆向き法線の最小和ノルム |
| 収束閾値 | 0.003 | vectorMatching収束条件 |
| 最大反復 | 200 | vectorMatching最大反復 |
| 減衰係数 | 0.3 | 変換適用の減衰 |
| 新規ノード距離 | 0.2 | マップ追加の最小距離 |

### ICP

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| 距離閾値 | 0.2 | 外れ値除去閾値 |

## 現在の実装状況

| 機能 | 実装済み | 説明 |
|------|:-------:|------|
| 法線計算 | ✓ | `_compute_normal_pca()` |
| サーフェス分類 | ✓ | `_classify_surface_type()` |
| 安定性追跡 | ✓ | `stability_age` |
| 自動注目検出 | ✓ | `auto_attention` |
| グローバルマップ管理 | - | `mapData[0]` 相当 |
| Surface Matching | - | `surfaceMatching()` |
| ICP | - | `icp()` |
| 座標系変換 | - | `getGNGdata()` |

## 実装に必要な追加コンポーネント

SLAM機能を実装する場合：

1. **MapManager クラス**
   - グローバルマップ（ノード + エッジ）管理
   - ノード追加/削除
   - 年齢管理

2. **SurfaceMatching クラス**
   - 特徴ノード選択
   - 対応点探索
   - 変換推定（法線ベース）

3. **ICP クラス**
   - 対応点探索
   - Horn法による変換推定

4. **SLAMState**
   - 累積位置・回転
   - 座標変換

## 参考文献

- Horn, B. K. (1987). "Closed-form solution of absolute orientation using unit quaternions." JOSA A, 4(4), 629-642.
- Saputra, A.A., et al. (2019). "Dynamic Density Topological Structure Generation for Real-Time Ladder Affordance Detection." IROS 2019.

## 関連ファイル

### リファレンス実装
- `references/original_code/azhar_ddgng/DepthSensor_Buggy/surfMatching.cpp`
- `references/original_code/azhar_ddgng/DepthSensor_Buggy/gng.cpp`
- `references/original_code/azhar_ddgng/DepthSensor_Buggy/gng.hpp`
- `references/original_code/toda_gngdt/gng_livox/src/icp.c`
- `references/original_code/toda_gngdt/gng_livox/src/icp.h`

### 現在の実装
- `algorithms/dd_gng/python/model.py`
- `algorithms/dd_gng/cpp/dd_gng.hpp`
