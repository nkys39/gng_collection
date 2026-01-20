//
//  GrowingNeuralGas.hpp
//
//  Created by watanabe on 2024/05/16.
//

#ifndef GrowingNeuralGas_hpp
#define GrowingNeuralGas_hpp

#include <iostream>
#include <stdio.h>
#include <vector>
#include <queue>
#include <unordered_set>
#include <unordered_map>

#define EIGEN_NO_DEBUG // コード内のassertを無効化
#include <Eigen/Core>
#include <Eigen/Eigen>

#define RUNTIME_PER_FRAME   100    // １フレームの学習回数
#define MAX_NODE_NUM        100    // 最大ノード数
#define START_NODE_NUM      2       // 初期ノード数
#define lAMBDA              100     // ノード追加周期

#define LEARNRATE_S1        0.08    // 学習係数
#define LEARNRATE_S2        0.008   // 学習係数
#define BETA                0.005   // 減衰率
#define ALFA                0.5     // 減衰率

#define MAX_EDGE_AGE        100     // 最大エッジ年齢

namespace GNG {

/* ノードの特徴量 */
struct Status {
    //!< ノードに付与したい特徴量
    Status(){};
};

/* ノードクラス */
template <typename PointT>
class NeuronNode {
public:

    int id = -1;        //!< ID
    float error = 1.0;  //!< 積算誤差

    /* ベクトル */
    PointT weight;

    /* 特徴量 */
    Status status;

    NeuronNode(){};
    NeuronNode(int id, PointT weight) : id(id), weight(weight) {};
    ~NeuronNode(){};
};

template <typename PointT>
class GrowingNeuralGas {
    /* データ構造 */
public:
    NeuronNode<PointT> nodes[MAX_NODE_NUM];                             //!< ノードリスト
    std::unordered_map<int, std::unordered_set<int>> edges_per_node;    //!< 隣接ノードの集合 key: node id, value: neighbours
    Eigen::MatrixXi edges = Eigen::MatrixXi::Zero(MAX_NODE_NUM, MAX_NODE_NUM); //!< エッジの集合(エッジ年齢) 0:非接続, ≥1:接続
    int n_learning = 0; //!< 学習回数カウント

private:
    std::queue<int> addable_node_indicies;  //!< 追加可能なノードID
    int n_trial = 0;  //!< 試行回数
    int weight_demention = 2;

    /* 関数 */
public:
    GrowingNeuralGas(int weight_demention);
    ~GrowingNeuralGas();

    /* GNG 実行関数 */
    void gngTrain(const std::vector<PointT>& sample_data);

private:
    /* neuron 操作関数 */
    void update_node_weight(int node_id, PointT weight, float step);    // ノードの重みの更新
    int add_node(PointT weight);                                        // ノードの追加
    void remove_node(int node_id);                                          // ノードの削除
    void add_edge(int node_1, int node_2);                                  // エッジの追加
    void remove_edge(int node_1, int node_2);                               // エッジの削除
    float calc_squaredNorm(PointT weight_1, PointT weight_2);       // 距離(二乗誤差)計算

    /* 学習関数 */
    void one_train_update(PointT sample);

};

}

#endif /* GrowingNeuralGas_hpp */

/* **********************************************************************

"""
 ノード情報 (NeuronNode)
 ----------
    id :
        ノードID（= nodes[]の添字）
        無効なノードは、id = -1 とする

    error :
        積算誤差

    weight :
        ノードのベクトル
        型は、Eigenの1次元配列
            <-- typedef Eigen::Matrix<float, DIMENSION, 1> WeightType;
        次元は、``DIMENSION``の値から指定可能

    status :
        特徴量管理用の構造体
        学習とは直接関係ない

    NeuronNode(``id``, ``WeightType(x,x,...,x)``) で作成

 無向グラフ構造
 ----------
 edges_per_node :
     各ノードをキーとし、そのノードに接続するノード（隣接ノード）のリストを値として持つ連想配列
     key: node id, value: neighbour node idicies

 edges :
     各接続ごとの年齢を格納する隣接配列

 nodes :
     グラフ内のノードのリスト
     ノードの情報は、構造体``NeuronNode``が持つ

 学習管理用変数
 ----------
 addable_node_indicies :
     追加可能な空きノードのIDを管理するキュー
     .push(id)で末尾に追加
     .front()で先頭のIDを取り出し、取り出した後は.pop()で先頭を削除

     n_trial :
     学習の試行回数
     ノード追加時に使用

 サンプルデータの形式
 ----------
    std::vector<WeightType> sample_data :
        型は、WeightTypeの動的配列
        Eigenの1次元配列
            <-- ``typedef Eigen::Matrix<float, DIMENSION, 1> WeightType;``
        次元は、``DIMENSION``の値で指定
        GNG実行関数``gngTrain(sample_data)``の引数にする

"""
***********************************************************************/
