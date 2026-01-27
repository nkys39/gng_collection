#pragma once
#include <iostream>

#include <utility>
#include <set>
#include <unordered_set>
#include <boost/heap/pairing_heap.hpp>
#include <Eigen/Dense>

int winner_count_bf = 0;
int winner1_count_nsw = 0;
int total_count_nsw = 0;
int wrong_count = 0;
bool wrong_flag = false;

// オブジェクト指向のGNGアルゴリズム
namespace GNG
{
    struct Node;
    struct Edge;
    struct Winners;
    struct HNSWNode;

    // Node : GNGアルゴリズムのノードを表す構造体
    struct Node
    {
        Eigen::VectorXd reference_vector; // 参照ベクトル
        double purity;                    // ノードの純度
        Eigen::MatrixXd vcm;              // ノードの分散共分散行列
        double squared_radius;            // 分散共分散行列の行列式
        Eigen::MatrixXd inv_vcm;          // 分散共分散行列の逆行列

        std::vector<std::pair<Node *, Edge *>> neighbors; // 隣接ノードとエッジのセット（全要素舐める場合mapは遅いのでvectorで実装）
        HNSWNode *hnsw_node;                              // 対応するHNSWノード
    };

    // Edge : GNGアルゴリズムのエッジを表す構造体
    struct Edge
    {
        double length; // エッジの長さ（未使用）
        double weight; // エッジの重み
    };

    // Winners : 勝者ノードとその誤差を保持する構造体
    struct Winners
    {
        Node *s1;        // 第一勝者ノード
        Node *s2;        // 第二勝者ノード
        double error_s1; // 第一勝者の誤差
        double error_s2; // 第二勝者の誤差
    };

    // ノードの積算誤差を比較する関数
    class CompareNodesByError
    {
    public:
        bool operator()(Node *a, Node *b) const
        {
            return a->squared_radius < b->squared_radius;
        }
    };

    struct HNSWNode // HNSWノードのクラス
    {
        Node *gng_node;            // 対応するGNGノード
        int distance;              // 検索時の距離
        double euclidean_distance; // 入力ベクトルとのユークリッド距離
        int time_stamp;            // ノード追加時のタイムスタンプ
        double level;              // ノードの連続レベル
    };

    // HNSWノードを距離で比較する関数（大きい順）
    class CompareHNSWNodesGreater
    {
    public:
        bool operator()(HNSWNode *a, HNSWNode *b) const
        {
            return a->euclidean_distance > b->euclidean_distance;
        }
    };

    // HNSWノードを距離で比較する関数（小さい順）
    class CompareHNSWNodesLess
    {
    public:
        bool operator()(HNSWNode *a, HNSWNode *b) const
        {
            return a->euclidean_distance < b->euclidean_distance;
        }
    };

    // HNSW : Hierarchical Navigable Small World に基づく近傍探索のクラス
    class HNSW
    {
    public:
        int ef_upper = 2;  // 上層検索時のビーム幅
        int ef_lower = 10; // 下層検索時のビーム幅
        double mL = 0.5;   // 上層遷移確率パラメータ

        HNSWNode *entry_node = nullptr;  // エントリーポイントノード
        std::set<HNSWNode *> hnsw_nodes; // HNSWノードのセット
        int node_count;                  // ノード数
        double log_node_count;           // ノード数の対数

        HNSW();
        ~HNSW();

        void insert(Node *gng_node);
        Winners hierarchicalBeamSearch(const Eigen::VectorXd &input_vector);
    };

    // GNG : GNGのクラス
    class GNG
    {
    public:
        int new_node_interval = 300;         // 新しいノードを追加する間隔(lambda)
        int max_nodes = 100;                 // 最大ノード数（メモリ依存）
        double eta1 = 0.05;                  // 第一勝者ノードの学習率
        double eta2 = 0.006;                 // 第一勝者ノードの隣接ノードの学習率
        double node_removal_threshold = 2;   // ノード削除の閾値
        double edge_removal_threshold = 0.5; // エッジ削除の閾値

        Node *min_purity_node;                   // 最小純度ノード
        std::set<Node *> nodes;                  // ノードのリスト
        std::unordered_set<Node *> cached_nodes; // キャッシュ用ノードセット
        HNSW hnsw;                               // HNSWインスタンス
        int step;                                // ステップ数
        int cycle;                               // サイクル数

        GNG();
        ~GNG();

        void initialize(const Eigen::VectorXd &input_vector1, const Eigen::VectorXd &input_vector2, const Eigen::VectorXd &min_coords, const Eigen::VectorXd &max_coords);
        Winners selectWinners(const Eigen::VectorXd &input_vector);
        bool checkEdgeExists(Node *node);
        bool updateEdges(Winners &winners);
        void updateReferenceVectors(Winners &winners, const Eigen::VectorXd &input_vector);
        std::pair<Node *, Node *> findNodesUandF();
        void addNode(Node *max_error_node, Node *max_error_neighbor);
        void addIfSilent(Winners &winners, const Eigen::VectorXd &input_vector);
        void removeNodes(Node *node);
        void calculateMinPurity();
        void run(const Eigen::VectorXd &input_vector_with_label);
    };

    // HNSW の実装 -------------------------------------------------
    // コンストラクタ
    HNSW::HNSW()
        : node_count(0), log_node_count(0.0)
    {
    }

    // デストラクタ
    HNSW::~HNSW()
    {
        // HNSWノードのメモリを解放
        for (auto node : hnsw_nodes)
        {
            delete node;
        }
    }

    // ノードをHNSWに挿入
    void HNSW::insert(Node *gng_node)
    {
        HNSWNode *new_hnsw_node = new HNSWNode();
        // HNSWノードとGNGノードを相互に関連付ける
        new_hnsw_node->gng_node = gng_node;
        gng_node->hnsw_node = new_hnsw_node;

        // HNSWノードの初期化
        new_hnsw_node->distance = -1;
        new_hnsw_node->euclidean_distance = -1.0;
        new_hnsw_node->time_stamp = node_count++;
        new_hnsw_node->level = std::log(1 + new_hnsw_node->time_stamp);

        // レベル計算用のノード数対数を更新
        log_node_count = std::log(1 + node_count);

        hnsw_nodes.insert(new_hnsw_node);
    }

    // 階層的ビームサーチによる近傍ノードの探索
    Winners HNSW::hierarchicalBeamSearch(const Eigen::VectorXd &input_vector)
    {
        // エントリーポイントノードの初期化
        entry_node->distance = 0;
        entry_node->euclidean_distance = (entry_node->gng_node->reference_vector - input_vector).squaredNorm();

        // 候補者ヒープと結果ヒープを作成
        boost::heap::pairing_heap<HNSWNode *, boost::heap::compare<CompareHNSWNodesGreater>> candidate_heap;
        boost::heap::pairing_heap<HNSWNode *, boost::heap::compare<CompareHNSWNodesLess>> result_heap;

        // フラグリセット用ノードリスト
        std::set<HNSWNode *> visited_nodes;

        // エントリーポイントノードを候補者ヒープに追加
        candidate_heap.push(entry_node);
        visited_nodes.insert(entry_node);

        // デバッグ用のルート表示用ベクター
        std::vector<HNSWNode *> debug_search_path;

        // 初期レベルの設定
        int current_level = static_cast<int>(std::floor((log_node_count - entry_node->level) * mL));

        // ビーム幅の設定
        int beam_width;
        if (current_level == 0)
            beam_width = ef_lower;
        else
            beam_width = ef_upper;

        // 階層的ビームサーチのメインループ
        HNSWNode *current_node;
        while (true)
        {
            if (candidate_heap.empty())
            {
                if (current_level == 0)
                    break;
                else
                {
                    // レベルを一つ下げる
                    current_level--;
                    if (current_level == 0)
                        beam_width = ef_lower;

                    // 上層での勝者ノードを取得
                    Node *first_winner_gng_node = nullptr;
                    Node *second_winner_gng_node = nullptr;

                    // result_heapをcandidate_heapに移動
                    while (!result_heap.empty())
                    {
                        if (result_heap.size() == 2)
                        {
                            first_winner_gng_node = result_heap.top()->gng_node;
                        }
                        else if (result_heap.size() == 1)
                        {
                            second_winner_gng_node = result_heap.top()->gng_node;
                        }
                        candidate_heap.push(result_heap.top());
                        result_heap.pop();
                    }

                    // 上層でもGNGの学習則を適用して勝者ノード間にエッジを追加
                    if (first_winner_gng_node != nullptr && second_winner_gng_node != nullptr)
                    {
                        bool edge_exists = false;
                        for (auto [neighbor, edge] : first_winner_gng_node->neighbors)
                        {
                            if (neighbor == second_winner_gng_node)
                            {
                                edge_exists = true;
                                break;
                            }
                        }
                        if (!edge_exists)
                        {
                            Edge *new_edge = new Edge{0.0, 0.0};
                            first_winner_gng_node->neighbors.push_back({second_winner_gng_node, new_edge});
                            second_winner_gng_node->neighbors.push_back({first_winner_gng_node, new_edge});
                        }
                    }
                    // 下層で探索続行
                    continue;
                }
            }

            // 候補者ヒープから最小距離ノードを取得
            current_node = candidate_heap.top();
            candidate_heap.pop();

            // デバッグ用に探索パスに追加
            debug_search_path.push_back(current_node);

            // 結果ヒープがビーム幅未満なら追加、そうでなければ最大距離ノードと比較して更新
            if (result_heap.size() < beam_width)
            {
                result_heap.push(current_node);
            }
            else if (current_node->euclidean_distance < result_heap.top()->euclidean_distance)
            {
                result_heap.pop();
                result_heap.push(current_node);
            }
            else
            {
                // 現在のノードが結果ヒープの最大距離よりも遠い場合、このレベルでの探索を終了
                candidate_heap.clear();
                continue;
            }

            // 隣接ノードを探索
            for (auto [gng_neighbor, edge] : current_node->gng_node->neighbors)
            {
                HNSWNode *neighbor = gng_neighbor->hnsw_node;

                // レベル制限を考慮
                if (static_cast<int>(std::floor((log_node_count - neighbor->level) * mL)) < current_level)
                    continue;

                // 未探索ノードなら距離を計算して候補者ヒープに追加
                if (neighbor->distance == -1)
                {
                    neighbor->distance = current_node->distance + 1;
                    neighbor->euclidean_distance = (neighbor->gng_node->reference_vector - input_vector).squaredNorm();

                    candidate_heap.push(neighbor);
                    visited_nodes.insert(neighbor);
                }
            }
        }

        // 探索フラグをリセット
        for (auto node : visited_nodes)
        {
            node->distance = -1;
        }
        visited_nodes.clear();

        // 結果ヒープから第一、第二勝者ノードを取得
        while (result_heap.size() > 2)
        {
            result_heap.pop();
        }
        Winners winners;
        winners.s2 = result_heap.top()->gng_node;
        winners.error_s2 = result_heap.top()->euclidean_distance;
        result_heap.pop();
        winners.s1 = result_heap.top()->gng_node;
        winners.error_s1 = result_heap.top()->euclidean_distance;

        // デバッグ用
        if (winners.error_s1 > 0.5 && node_count > 30)
        {
            // std::cout << "Input vector:" << std::endl;
            // std::cout << input_vector.transpose() << std::endl;
            // std::cout << std::endl << std::endl;
            // for (auto node : debug_search_path)
            // {
            //     std::cout << node->gng_node->reference_vector.transpose() << std::endl;
            // }
            std::cout << "Beam search winner error: " << winners.error_s1 << std::endl;
            if (!wrong_flag)
            {
                wrong_count++;
                wrong_flag = true;
            }
        }

        return winners;
    }
    // HNSW の実装ここまで ---------------------------------------------

    // GNG の実装 -------------------------------------------------
    // コンストラクタ
    GNG::GNG()
        : step(0), cycle(0)
    {
    }

    // デストラクタ
    GNG::~GNG()
    {
        std::set<Edge *> all_edges;
        // ノードとエッジのメモリを解放
        for (auto node : nodes)
        {
            for (auto [neighbor, edge] : node->neighbors)
            {
                all_edges.insert(edge); // エッジをセットに追加
            }
            // ノードのメモリを解放
            delete node;
        }
        for (auto edge : all_edges)
        {
            // エッジのメモリを解放
            delete edge;
        }
    }

    // 初期化
    void GNG::initialize(const Eigen::VectorXd &input_vector_with_label1, const Eigen::VectorXd &input_vector_with_label2, const Eigen::VectorXd &min_coords, const Eigen::VectorXd &max_coords)
    {
        // ラベルの除去
        Eigen::VectorXd input_vector1 = input_vector_with_label1.head(input_vector_with_label1.size() - 1);
        Eigen::VectorXd input_vector2 = input_vector_with_label2.head(input_vector_with_label2.size() - 1);

        double squared_distance = (input_vector1 - input_vector2).squaredNorm();

        std::set<Edge *> all_edges;
        // ノードとエッジのメモリを解放
        for (auto node : nodes)
        {
            for (auto [neighbor, edge] : node->neighbors)
            {
                all_edges.insert(edge); // エッジをセットに追加
            }
            // ノードのメモリを解放
            delete node;
        }
        for (auto edge : all_edges)
        {
            // エッジのメモリを解放
            delete edge;
        }
        nodes.clear();

        // 分散共分散行列の初期化
        // イプシロンをd^2/100に設定（ノード間を分散10で分離するイメージ）
        Eigen::MatrixXd initial_vcm_inv = 36 / squared_distance * Eigen::MatrixXd::Identity(input_vector1.size(), input_vector1.size()); // 逆行列は初期値として大きな値の単位行列
        Eigen::MatrixXd initial_vcm = initial_vcm_inv.inverse();
        double initial_squared_radius = initial_vcm.trace(); // 行列式の初期値
        // double initial_squared_radius = squared_distance;

        // 初期ノードの追加
        Node *initial_node1 = new Node{
            input_vector1,          // 参照ベクトルは入力ベクトル1
            1.0,                    // 純度は初期値1.0
            initial_vcm,            // 分散共分散行列は逆行列の逆行列で初期化
            initial_squared_radius, // 行列式の初期値
            initial_vcm_inv,        // 逆行列の初期値はノード間距離の2乗×単位行列
            {},                     // 隣接ノードは空
            nullptr                 // 対応するHNSWノードはnullptrで初期化
        };
        Node *initial_node2 = new Node{
            input_vector2,
            1.0,
            initial_vcm,
            initial_squared_radius,
            initial_vcm_inv,
            {},
            nullptr};
        nodes.insert(initial_node1);
        nodes.insert(initial_node2);

        hnsw.node_count = 0; // HNSWノード数を初期化
        // HNSWへのノード挿入
        // GNGのエッジを使うため、二つ目のノードは後で追加
        hnsw.insert(initial_node1);
        hnsw.entry_node = initial_node1->hnsw_node;

        // 初期エッジの追加
        Edge *initial_edge = new Edge{0.0, 1.0};
        initial_node1->neighbors.push_back({initial_node2, initial_edge});
        initial_node2->neighbors.push_back({initial_node1, initial_edge});

        // HNSWへの二つ目のノード挿入
        hnsw.insert(initial_node2);

        step = 0;  // ステップ数を初期化
        cycle = 0; // サイクル数を初期化
    }

    // 勝者ノードを選択
    Winners GNG::selectWinners(const Eigen::VectorXd &input_vector)
    {
        Winners winners;
        winners.s1 = nullptr;
        winners.s2 = nullptr;
        winners.error_s1 = std::numeric_limits<double>::max();
        winners.error_s2 = std::numeric_limits<double>::max();

        // // HNSWで近傍ノードを探索
        // Winners winners_hnsw = hnsw.hierarchicalBeamSearch(input_vector);

        // 勝者ノードはユークリッド距離で選択（マハラノビス距離ではない）
        for (auto node : nodes)
        {
            double squared_error = (input_vector - node->reference_vector).squaredNorm();
            // node->squared_radius = node->vcm.trace(); // 分散共分散行列のトレースを使用
            // double squared_error = (input_vector - node->reference_vector).squaredNorm() / node->squared_radius;
            // // やっぱりマハラノビス距離で選ぶ
            // Eigen::VectorXd error_vector = input_vector - node->reference_vector;
            // node->inv_vcm = node->vcm.inverse();
            // double squared_error = error_vector.transpose() * node->inv_vcm * error_vector;
            if (squared_error < winners.error_s1)
            {
                winners.s2 = winners.s1;
                winners.error_s2 = winners.error_s1;
                winners.s1 = node;
                winners.error_s1 = squared_error;
            }
            else if (squared_error < winners.error_s2 && node != winners.s1)
            {
                winners.s2 = node;
                winners.error_s2 = squared_error;
            }
        }

        // if (nodes.size() >= 0)
        // {
        //     winner_count_bf++;
        //     if (winners.s1 == winners_hnsw.s1)
        //     {
        //         winner1_count_nsw++;
        //         if (winners.s2 == winners_hnsw.s2)
        //         {
        //             total_count_nsw++;
        //         }
        //     }
        //     else
        //     {
        //         if (winners_hnsw.error_s1 - winners.error_s1 > 0.5)
        //         {
        //             std::cout << "Difference in second winner error: " << winners_hnsw.error_s1 - winners.error_s1 << std::endl;
        //             hnsw.beamSearch(input_vector);
        //         }
        //     }
        // }

        // return winners_hnsw;
        return winners;

        // beam searchを使って勝者ノードを選択
        // std::set<std::pair<Node *, double>> candidate_nodes; // ノードとそのユークリッド距離のセット
    }

    // 孤立ノードのチェックと削除
    bool GNG::checkEdgeExists(Node *node)
    {
        bool has_edge = false;
        for (auto [neighbor, edge] : node->neighbors)
        {
            if (edge->weight >= edge_removal_threshold)
            {
                has_edge = true;
                break;
            }
        }

        if (!has_edge)
        {
            for (auto [neighbor, edge] : node->neighbors)
            {
                // 隣接ノードからエッジを削除
                auto &neighbor_edges = neighbor->neighbors;
                neighbor_edges.erase(std::remove_if(neighbor_edges.begin(), neighbor_edges.end(),
                                                    [node](const std::pair<Node *, Edge *> &pair)
                                                    { return pair.first == node; }),
                                     neighbor_edges.end());
                delete edge; // エッジのメモリを解放
            }
            node->neighbors.clear();
            nodes.erase(node); // ノードを削除
            cached_nodes.erase(node);
            delete node; // メモリを解放
        }
        return has_edge;
    }

    // エッジを更新
    bool GNG::updateEdges(Winners &winners)
    {
        bool edge_exists = false;
        // std::set<Edge *> copy_edges = winners.s1->edges; // エッジの削除でイテレータが無効になるのを防ぐためにコピーを作成
        std::vector<std::pair<Node *, Edge *>> copy_neighbors = winners.s1->neighbors; // エッジの削除でイテレータが無効になるのを防ぐためにコピーを作成
        for (auto [neighbor, edge] : copy_neighbors)
        {
            if (neighbor == winners.s2) // 第二勝者ノードが隣接ノードの場合
            {
                edge->weight = 1.0; // エッジの重みをリセット
                // edge->weight += eta1 * (1.0 - edge->weight);
                edge_exists = true; // エッジが存在する場合
            }
            else
            {
                edge->weight += eta2 * (0.0 - edge->weight);
            }

            // エッジの重みが閾値以下の場合、隣接が孤立している可能性があるためチェック
            if (edge->weight < edge_removal_threshold)
            {
                checkEdgeExists(neighbor);
            }
        }

        // 第一勝者ノードが孤立しているかチェック
        bool s1_exists = checkEdgeExists(winners.s1);

        if (!edge_exists && s1_exists)
        {
            // 新しいエッジを作成
            Edge *new_edge = new Edge{0.0, 1.0};
            winners.s1->neighbors.push_back({winners.s2, new_edge}); // ノード1にエッジを追加
            winners.s2->neighbors.push_back({winners.s1, new_edge}); // ノード2にエッジを追加
        }
        return s1_exists;
    }

    // 参照ベクトルを更新
    void GNG::updateReferenceVectors(Winners &winners, const Eigen::VectorXd &input_vector)
    {
        // 第一勝者ノードの参照ベクトルを更新
        Eigen::VectorXd error_vector = input_vector - winners.s1->reference_vector;
        winners.s1->reference_vector += eta1 * error_vector;

        // 第一勝者ノードの分散共分散行列を更新
        winners.s1->vcm += eta1 * (error_vector * error_vector.transpose() - winners.s1->vcm);
        // winners.s1->squared_radius += eta1 * (winners.error_s1 - winners.s1->squared_radius);
        cached_nodes.insert(winners.s1);

        // // 第二勝者ノードの共分散行列も更新
        // error_vector = input_vector - winners.s2->reference_vector;
        // winners.s2->vcm += eta1 * (error_vector * error_vector.transpose() - winners.s2->vcm);
        // cached_nodes.insert(winners.s2);

        // 隣接ノードの参照ベクトルを更新
        for (auto [neighbor, edge] : winners.s1->neighbors)
        {
            if (edge->weight < edge_removal_threshold)
                continue; // エッジが削除されている場合はスキップ

            error_vector = input_vector - neighbor->reference_vector;
            // double squared_error = error_vector.squaredNorm();

            neighbor->reference_vector += eta2 * edge->weight * error_vector;
            // 隣接ノードの分散共分散行列を更新
            neighbor->vcm += eta2 * edge->weight * (error_vector * error_vector.transpose() - neighbor->vcm);
            // neighbor->squared_radius += eta2 * edge->weight * (error_vector.squaredNorm() - neighbor->squared_radius);
            cached_nodes.insert(neighbor);
        }
    }

    // 最大の誤差を持つノードuとその隣接ノードfを見つける
    std::pair<Node *, Node *> GNG::findNodesUandF()
    {
        if (cached_nodes.empty())
        {
            return {nullptr, nullptr};
        }
        // 最大の誤差を持つノードuを見つける
        Node *max_error_node = nullptr;
        for (auto node : cached_nodes)
        {
            // 共分散行列の楕円体の面積を積算誤差として使用
            node->squared_radius = node->vcm.trace();
            if (max_error_node == nullptr || node->squared_radius > max_error_node->squared_radius)
            {
                max_error_node = node;
            }
        }
        // 最大の誤差を持つノードの隣接ノードの中から最も誤差が大きいノードを見つける（f）
        Node *max_error_neighbor = nullptr;
        double max_squared_radius = 0.0;
        for (auto [neighbor, edge] : max_error_node->neighbors)
        {
            if (edge->weight < edge_removal_threshold)
                continue; // エッジが削除されている場合はスキップ
            if (max_error_neighbor == nullptr || neighbor->squared_radius * edge->weight > max_squared_radius)
            {
                max_squared_radius = neighbor->squared_radius * edge->weight;
                max_error_neighbor = neighbor;
            }
        }

        return {max_error_node, max_error_neighbor};
    }

    // ノードuとノードfの中点を新しいノードとして追加
    void GNG::addNode(Node *max_error_node, Node *max_error_neighbor)
    {
        max_error_node->vcm *= 0.25;
        max_error_neighbor->vcm *= 0.25;
        max_error_node->squared_radius *= 0.25;
        max_error_neighbor->squared_radius *= 0.25;
        max_error_node->inv_vcm *= 4.0;
        max_error_neighbor->inv_vcm *= 4.0;

        // 新しいVCMの計算
        Eigen::MatrixXd new_vcm = 0.5 * (max_error_node->vcm + max_error_neighbor->vcm);
        Eigen::MatrixXd new_inv_vcm = new_vcm.inverse();
        double new_squared_radius = new_vcm.trace();
        // double new_squared_radius = 0.5 * (max_error_node->squared_radius + max_error_neighbor->squared_radius);

        // uとfの中点を新しいノードとして追加
        Eigen::VectorXd new_reference_vector = 0.5 * (max_error_node->reference_vector + max_error_neighbor->reference_vector);
        Node *new_node = new Node{
            new_reference_vector,
            0.5 * (max_error_node->purity + max_error_neighbor->purity),
            new_vcm,
            new_squared_radius,
            new_inv_vcm,
            {},
            nullptr};
        nodes.insert(new_node);

        // uとfを繋ぐエッジを削除
        for (auto [neighbor, edge] : max_error_node->neighbors)
        {
            if (neighbor == max_error_neighbor)
            {
                edge->weight = 0.0; // エッジの重みを0に設定（削除の代わり）
                break;
            }
        }
        // 新しいノードとu、fを繋ぐエッジを追加
        Edge *edge1 = new Edge{0.0, 1.0};                           // 新しいエッジの年齢と長さは0
        Edge *edge2 = new Edge{0.0, 1.0};                           // 新しいエッジの年齢と長さは0
        new_node->neighbors.push_back({max_error_node, edge1});     // 新しいノードからuへのエッジを追加
        max_error_node->neighbors.push_back({new_node, edge1});     // uから新しいノードへのエッジを追加
        new_node->neighbors.push_back({max_error_neighbor, edge2}); // 新しいノードからfへのエッジを追加
        max_error_neighbor->neighbors.push_back({new_node, edge2}); // fから新しいノードへのエッジを追加

        // HNSWへのノード挿入
        hnsw.insert(new_node);
    }

    // Add If Silentに基づくノード追加
    void GNG::addIfSilent(Winners &winners, const Eigen::VectorXd &input_vector)
    {
        if (nodes.size() == max_nodes)
        {
            return; // ノード数が最大に達している場合、追加しない
        }
        // // mahalanobis距離を用いてノード追加を判定
        // Eigen::VectorXd error_vector = input_vector - winners.s1->reference_vector;
        // winners.s1->inv_vcm = winners.s1->vcm.inverse();
        // double mahalanobis_distance = std::sqrt(error_vector.transpose() * winners.s1->inv_vcm * error_vector);
        // double threshold = 2.0; // 閾値を適宜設定

        // traceを用いてノード追加を判定
        double euclidean_distance = (input_vector - winners.s1->reference_vector).squaredNorm();
        double threshold = 4.0 * winners.s1->squared_radius; // 閾値を適宜設定

        if (euclidean_distance > threshold)
        {
            Node *new_node = new Node{
                input_vector,
                1.0,
                winners.s1->vcm,
                winners.s1->squared_radius,
                winners.s1->inv_vcm,
                {},
                nullptr};
            nodes.insert(new_node);

            // 第一勝者ノードとのエッジを追加
            Edge *new_edge = new Edge{0.0, 1.0};                   // 新しいエッジの年齢と長さは0
            new_node->neighbors.push_back({winners.s1, new_edge}); // 新しいノードから第一勝者ノードへのエッジを追加
            winners.s1->neighbors.push_back({new_node, new_edge}); // 第一勝者ノードから新しいノードへのエッジを追加

            // HNSWへのノード挿入
            hnsw.insert(new_node);
        }
        return;
    }

    // ユーティリティ値が閾値を超えたノードを削除
    void GNG::removeNodes(Node *node)
    {
        for (auto [neighbor, edge] : node->neighbors)
        {
            // 隣接ノードとのエッジを削除
            neighbor->neighbors.erase(std::remove_if(neighbor->neighbors.begin(), neighbor->neighbors.end(),
                                                     [node](const std::pair<Node *, Edge *> &p)
                                                     { return p.first == node; }),
                                      neighbor->neighbors.end());
            // ノードが孤立している場合、ノードを削除
            if (neighbor->neighbors.empty())
            {
                nodes.erase(neighbor); // ノード2を削除
                cached_nodes.erase(neighbor);
                delete neighbor; // メモリを解放
            }
            delete edge; // エッジのメモリを解放
        }
        node->neighbors.clear();
        nodes.erase(node); // ノード1を削除
        cached_nodes.erase(node);
        delete node; // メモリを解放
    }

    void GNG::calculateMinPurity()
    {
        min_purity_node = nullptr;
        double min_purity = 1.0;
        for (auto node : nodes)
        {
            if (min_purity_node == nullptr || node->purity < min_purity)
            {
                min_purity_node = node;
                min_purity = node->purity;
            }
        }
    }

    // 1ステップの実行
    void GNG::run(const Eigen::VectorXd &input_vector_with_label)
    {
        // ラベルを除去
        Eigen::VectorXd input_vector = input_vector_with_label.head(input_vector_with_label.size() - 1);
        int label = static_cast<int>(input_vector_with_label(input_vector_with_label.size() - 1));
        Winners winners = selectWinners(input_vector);

        if (label == -1)
        {
            Eigen::VectorXd error_vector = input_vector - winners.s1->reference_vector;
            // 誤差が誤差半径の倍以上の場合、mahalanobis距離を計算して純度を更新
            if (error_vector.squaredNorm() > 4.0 * winners.s1->squared_radius)
            {
                winners.s1->inv_vcm = winners.s1->vcm.inverse();
                double mahalanobis_distance = std::sqrt(error_vector.transpose() * winners.s1->inv_vcm * error_vector);
                if (mahalanobis_distance < 1.0)
                {
                    winners.s1->purity += eta1 * (1.0 - mahalanobis_distance) * (0.0 - winners.s1->purity);
                }
            }

            for (auto [neighbor, edge] : winners.s1->neighbors)
            {
                edge->weight += eta2 * (0.0 - edge->weight);
                // s1とs2にエッジがある場合、エッジの重みを更新
                if (neighbor == winners.s2)
                {
                    edge->weight += eta1 * (0.0 - edge->weight);
                    // エッジが孤立したら削除
                    if (edge->weight < edge_removal_threshold)
                    {
                        checkEdgeExists(winners.s1);
                        checkEdgeExists(winners.s2);
                    }
                    break;
                }
            }
        }
        else
        {
            winners.s1->purity += eta1 * (1.0 - winners.s1->purity);

            // Add If Silent に基づくノード追加
            // addIfSilent(winners, input_vector);

            // 勝者ノードのラベルが-1以外の場合、通常の処理を実行
            updateReferenceVectors(winners, input_vector);
            updateEdges(winners);
        }
        step++;

        // 最小純度ノードの計算
        // calculateMinPurity();
        // double min_purity = min_purity_node->purity;

        // utilityもどき
        // std::pair<Node *, Node *> nodes_u_f = findNodesUandF();
        // if (nodes.size() > 2 && nodes_u_f.first->error_radius > winners.s1->utility_radius * node_removal_threshold && min_purity == 1.0)
        // {
        //     removeNodes(winners.s1);
        // }

        // ノード数の動的調整
        if (step % new_node_interval == 0)
        {

            // utilityもどき
            // std::pair<Node *, Node *> nodes_u_f = findNodesUandF();
            // if (nodes.size() > 2 && nodes_u_f.first->error_radius > winners.s1->utility_radius * node_removal_threshold && min_purity == 1.0)
            // {
            //     // removeNodes(winners.s1);
            // }
            // if (nodes.size() > 2 && min_purity < 1e-10)
            // {
            //     removeNodes(min_purity_node);
            // }

            // 新しいノードの追加
            // if (min_purity < (1.0 - 1e-5) && nodes.size() < max_nodes)
            if (nodes.size() < max_nodes)
            {
                std::pair<Node *, Node *> nodes_u_f = findNodesUandF();

                if (nodes_u_f.first != nullptr && nodes_u_f.second != nullptr)
                {
                    addNode(nodes_u_f.first, nodes_u_f.second);
                }
            }

            cached_nodes.clear();
            cycle++;
            step = 0;
        }
    }

    // GNG の実装ここまで ---------------------------------------------

}