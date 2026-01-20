//
//  GrowingNeuralGas.cpp
//
//  Created by watanabe on 2024/05/16.
//

#include "GrowingNeuralGas.hpp"

using namespace GNG;
using namespace std;

template <typename PointT>
GrowingNeuralGas<PointT>::GrowingNeuralGas(int weight_demention) : weight_demention(weight_demention)
{
    // キューに追加可能ノード登録
    for (int i = 0; i < MAX_NODE_NUM; i++) {
        addable_node_indicies.push(i);
    }
    // 初期ノード数に応じて、キューから取り出し、ランダムにノードを生成
    PointT random_weight;
    for (int i = 0; i < START_NODE_NUM; i++) {
        for (int j = 0; j < weight_demention; j++) {
            random_weight(j) = (float)rand()/RAND_MAX;
        }
        add_node(random_weight);
    }
}

template <typename PointT>
GrowingNeuralGas<PointT>::~GrowingNeuralGas()
{
    // メモリ解放
//    for (int i = 0; i < MAX_NODE_NUM; i++) {
//        delete[] edges[i];
//    }
//    delete[] edges;
}

/* ノード・エッジ処理関数 */
// ノードの重みの更新
template <typename PointT>
void GrowingNeuralGas<PointT>::update_node_weight(int node_id, PointT weight, float step)
{
    nodes[node_id].weight += step * (weight - nodes[node_id].weight);
}

// ノードの追加
template <typename PointT>
int GrowingNeuralGas<PointT>::add_node(PointT weight)
{
    int node_id = addable_node_indicies.front();    // 空きノードのID取得
    addable_node_indicies.pop();
    nodes[node_id] = NeuronNode<PointT>(node_id, weight);
    edges_per_node[node_id] = unordered_set<int>();
    return node_id;
}

// ノードの削除
template <typename PointT>
void GrowingNeuralGas<PointT>::remove_node(int node)
{
    if (!edges_per_node[node].empty()) return;
    edges_per_node.erase(node);
    nodes[node].id = -1;
    addable_node_indicies.push(node); // 削除したノードIDを追加可能ノードとして登録
}

// エッジの追加
template <typename PointT>
void GrowingNeuralGas<PointT>::add_edge(int node_1, int node_2)
{
    // すでにエッジが存在する場合、エッジ年齢をリセット
    if (edges(node_1, node_2) > 0) {
        edges(node_1, node_2) = 1;
        edges(node_2, node_1) = 1;
    }
    // エッジがまだない場合、接続
    else {
        edges_per_node[node_1].insert(node_2);
        edges_per_node[node_2].insert(node_1);
        edges(node_1, node_2) = 1;
        edges(node_2, node_1) = 1;
    }
}

// エッジの削除
template <typename PointT>
void GrowingNeuralGas<PointT>::remove_edge(int node_1, int node_2)
{
    edges_per_node[node_1].erase(node_2);
    edges_per_node[node_2].erase(node_1);
    edges(node_1, node_2) = 0;
    edges(node_2, node_1) = 0;
}

// 距離(二乗誤差)計算
template <typename PointT>
float GrowingNeuralGas<PointT>::calc_squaredNorm(PointT weight_1, PointT weight_2)
{
    return (weight_1 - weight_2).squaredNorm();
}

/* GNG 実行関数 */
template <typename PointT>
void GrowingNeuralGas<PointT>::gngTrain(const vector<PointT>& sample_data)
{
    int num_of_data = (int)sample_data.size();
    int ramdam_index;

    for (int k = 0; k < RUNTIME_PER_FRAME; k++)
    {
        while ((ramdam_index = (int)((double)num_of_data * rand()/RAND_MAX)) == num_of_data);
        one_train_update(sample_data[ramdam_index]);
    }
}

/* 学習関数 */
template <typename PointT>
void GrowingNeuralGas<PointT>::one_train_update(PointT sample)
{
    /* 第一、第二勝者ノード探索 */
    float min_dis1 = 1e9, min_dis2 = 1e9;  // 距離の仮保存
    int s1_id = -1, s2_id = -1;            // 第一勝者ノード, 第二勝者ノード
    static float dis;
    for (auto& node : nodes)
    {
        if (node.id == -1) continue;
        // すべてのノードの積算誤差を減らす
        node.error -= BETA * node.error;
        // 距離計算
        dis = calc_squaredNorm(sample, node.weight);
        if (dis < min_dis1) {
            min_dis2 = min_dis1; s2_id = s1_id;
            min_dis1 = dis; s1_id = node.id;
        }
        else if (dis < min_dis2) {
            min_dis2 = dis; s2_id = node.id;
        }
    }
    min_dis1 = sqrt(min_dis1);
//    min_dis2 = sqrt(min_dis2);

    /* ニューロン更新 */
    //- 第一勝者ノードの更新
    nodes[s1_id].error += min_dis1;
    update_node_weight(s1_id, sample, LEARNRATE_S1);
    //- s1,s2の接続
    add_edge(s1_id, s2_id);
    //- 第一勝者ノードの隣接ノードの更新 OR 古いエッジの削除、孤立ノードの削除
    std::vector<int> delete_edges; // エッジ削除用
    for (auto &neighbour_node_id : edges_per_node[s1_id])
    {
        if (edges(s1_id, neighbour_node_id) > MAX_EDGE_AGE) {
            delete_edges.push_back(neighbour_node_id);
        }
        else {
            update_node_weight(neighbour_node_id, sample, LEARNRATE_S2);
            edges(s1_id, neighbour_node_id)++;
            edges(neighbour_node_id, s1_id)++;
        }
    }
    for (auto to_node_id : delete_edges) {
        remove_edge(s1_id, to_node_id);
        if (edges_per_node[to_node_id].empty()) { // 孤立ノードの削除
            remove_node(to_node_id);
        }
    }

    /* 一定周期でノードの追加 */
    if (n_trial == lAMBDA) {
        n_trial = 0;
        //- ノード配列に空きがあるなら追加
        if (!addable_node_indicies.empty())
        {
            float max_err_q = 0.0, max_err_f = 0.0;     // 最大積算誤差の値
            int max_err_q_id = -1, max_err_f_id = -1;   // 最大積算誤差のノードID
            //- 最大積算誤差ノードの探索
            for (const auto& node : nodes) {
                if (node.id == -1) continue;
                if (max_err_q < node.error) {
                    max_err_q = node.error;
                    max_err_q_id = node.id;
                }
            }
            //- 隣接ノードの中から最大積算誤差ノードの探索
            for (const auto& node : edges_per_node[max_err_q_id]) {
                if (max_err_f < nodes[node].error) {
                    max_err_f = nodes[node].error;
                    max_err_f_id = nodes[node].id;
                }
            }
            //- ノードq,fを二分するノードの追加、エッジの追加
            static int new_node_id;
            new_node_id = add_node(PointT((nodes[max_err_q_id].weight + nodes[max_err_f_id].weight) * 0.5));
            remove_edge(max_err_q_id, max_err_f_id);
            add_edge(max_err_q_id, new_node_id);
            add_edge(max_err_f_id, new_node_id);
            //- ノードq,f,newの積算誤差更新
            nodes[max_err_q_id].error -= ALFA * nodes[max_err_q_id].error;
            nodes[max_err_f_id].error -= ALFA * nodes[max_err_f_id].error;
            nodes[new_node_id].error = (nodes[max_err_q_id].error + nodes[max_err_f_id].error) * 0.5;
        }
    }

    /* すべてのノードの積算誤差を減らす */
//    for (auto& node : nodes) {
//        if (node.id == -1) continue;
//        node.error -= BETA * node.error;
//    }
    n_trial++;
    n_learning++;

}

template class GNG::GrowingNeuralGas<Eigen::Vector2f>;
template class GNG::GrowingNeuralGas<Eigen::Vector3f>;
