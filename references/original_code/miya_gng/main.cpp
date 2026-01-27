#include <iostream>
#include <vector>
#include <Eigen/Dense>

#include <set>
#include <time.h>
// #include <opencv2/opencv.hpp>

#include "GNG.hpp"

int world_step = 0;

// サンプリング関数の定義
Eigen::VectorXd sampleInputVector()
{
    Eigen::VectorXd input_vector(3);
    // 0から1の範囲でランダムな値を生成
    double x = (static_cast<double>(rand()) / RAND_MAX - 0.5) * 4; // -2から2の範囲に変換
    double y = (static_cast<double>(rand()) / RAND_MAX - 0.5) * 4; // -2から2の範囲に変換

    // if (world_step >= 300000 && world_step < 300010)
    // {
    //     input_vector << x, y, -1.0;
    //     return input_vector;
    // }
    // double theta = static_cast<double>(rand()) / RAND_MAX * 2.0 * M_PI;
    // double radius = static_cast<double>(rand()) / RAND_MAX * 2.0;
    // x = radius * cos(theta);
    // y = radius * sin(theta);
    // input_vector << x, y, 0.0;
    // return input_vector;

    // if (world_step < 100000 || world_step > 500000)
    if (false)
    {
        while (true)
        {
            // 半径0.5から1.0の範囲でサンプリング
            if ((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) >= 0.5 && (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) <= 1.0)
            {
                input_vector << x - 1, y - 1, 0.0;
                break;
            }
            else if ((x + 1) * (x + 1) + (y + 1) * (y + 1) <= 0.25)
            {
                input_vector << x - 1, y - 1, 0.0;
                break;
            }
            // else if ((y + 1) * (y + 1) + (x - 1) * (x - 1) <= 0.16)
            // {
            //     input_vector << x, y, 0.0;
            //     break;
            // }
            // else if ((x + 1) * (x + 1) + (y - 1) * (y - 1) <= 0.16)
            // {
            //     input_vector << x, y, 0.0;
            //     break;
            // }
            // else if ((y+1)*(y+1) <= 0.01)
            // {
            //     if ((y+1)*(y+1) <= 0.001)
            //     {
            //         input_vector << x, y, 0.0;
            //     }
            //     else
            //     {
            //         input_vector << x, y, -1.0;
            //     }
            // }
            else
            {
                // // 境界線0.1の範囲でサンプリング
                // if ((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) <= 1.5)
                // {
                //     input_vector << x - 1, y - 1, -1.0;
                //     break;
                // }
                // else if ((x + 1) * (x + 1) + (y + 1) * (y + 1) <= 0.36)
                // {
                //     input_vector << x - 1, y - 1, -1.0;
                //     break;
                // }
                // else
                // {
                //     x = (static_cast<double>(rand()) / RAND_MAX - 0.5) * 4; // -2から2の範囲に変換
                //     y = (static_cast<double>(rand()) / RAND_MAX - 0.5) * 4; // -2から2の範囲に変換
                // }
                x = (static_cast<double>(rand()) / RAND_MAX - 0.5) * 4; // -2から2の範囲に変換
                y = (static_cast<double>(rand()) / RAND_MAX - 0.5) * 4; // -2から2の範囲に変換
            }
        }
    }else{
        while (true)
        {
            // L字型の形状（大きめに）
            if ((x >= -1.5 && x <= 1.5 && y >= -0.5 && y <= 0.5) || (x >= -0.5 && x <= 0.5 && y >= -1.5 && y <= 0.0))
            {
                input_vector << x, y, 0.0;
                break;
            }
            // 縦線
            else if (x >= -0.05 && x <= 0.05 && y >= -1.5 && y <= 1.5)
            {
                input_vector << x, y, 0.0;
                break;
            }
            else if (y >= 1.4 && y <= 1.5 && x >= -1.5 && x <= 1.5)
            {
                input_vector << x, y, 0.0;
                break;
            }
            else
            {
                // if ((x >= -1.6 && x <= 1.6 && y >= -0.6 && y <= 0.6) || (x >= -0.6 && x <= 0.6 && y >= -1.6 && y <= 0.1))
                // {
                //     input_vector << x, y, -1.0; // 境界線付近は重みを0に設定
                //     break;
                // }
                // else if (x >= -0.2 && x <= 0.2 && y >= -1.6 && y <= 1.6)
                // {
                //     input_vector << x, y, -1.0; // 境界線付近は重みを0に設定
                //     break;
                // }
                // else if (y >= 1.2 && y <= 1.6 && x >= -1.6 && x <= 1.6)
                // {
                //     input_vector << x, y, -1.0; // 境界線付近は重みを0に設定
                //     break;
                // }
                // else if ((x + 2) * (x + 2) + (y + 2) * (y + 2) <= 0.25)
                // {
                //     input_vector << x, y, -1.0; // 境界線付近は重みを0に設定
                //     break;
                // }
                // else
                // {
                //     x = (static_cast<double>(rand()) / RAND_MAX - 0.5) * 6; // -3から2の範囲に変換
                //     y = (static_cast<double>(rand()) / RAND_MAX - 0.5) * 6; // -3から2の範囲に変換
                // }
                x = (static_cast<double>(rand()) / RAND_MAX - 0.5) * 4; // -2から2の範囲に変換
                y = (static_cast<double>(rand()) / RAND_MAX - 0.5) * 4; // -2から2の範囲に変換
            }
        }
    }
    world_step++;
    return input_vector;
}

// // サンプリング関数の定義
// std::vector<Eigen::VectorXd> dataset;
// Eigen::VectorXd sampleInputVector()
// {
//     return dataset[rand() % dataset.size()];
// }




int main()
{
    for (int seed = 0; seed < 1; seed++)
    {
        std::cout << "Random Seed: " << seed << std::endl;
        world_step = 0;
        wrong_flag = false;
        // ランダムシードの初期化
        seed = 1;
        srand(static_cast<unsigned int>(seed));
        // // 画像を入力データセットとしてロード
        // auto img = cv::imread("distribution/sample_map.png", cv::IMREAD_GRAYSCALE);
        // if (img.empty())
        // {
        //     std::cerr << "Error: Could not load image." << std::endl;
        //     return -1;
        // }
        // // 画像の各ピクセルをデータセットに変換
        // for (int y = 0; y < img.rows; ++y)
        // {
        //     for (int x = 0; x < img.cols; ++x)
        //     {
        //         uchar pixel_value = img.at<uchar>(y, x);
        //         if (pixel_value == 255) // 白いピクセルのみを使用
        //         {
        //             Eigen::VectorXd vec(3);
        //             vec << x, 
        //                       y,
        //                    0.0; // 重み
        //             dataset.push_back(vec);
        //         }
        //         else if (pixel_value == 0)
        //         {
        //             Eigen::VectorXd vec(3);
        //             vec << x, 
        //                       y,
        //                    -1.0; // 重みを-1に設定
        //             dataset.push_back(vec);
        //         }
        //     }
        // }
        // std::cout << "Image: " << img.cols << " x " << img.rows << std::endl;
        // std::cout << "Dataset size: " << dataset.size() << std::endl;
        // std::cout << dataset[0].transpose() << std::endl;
        // std::cout << dataset[dataset.size() - 1].transpose() << std::endl;
        
        
        // GNGアルゴリズムのインスタンスを作成
        GNG::GNG gng;
        Eigen::VectorXd min_coords(2);
        Eigen::VectorXd max_coords(2);
        // min_coords << 0, 0;
        // max_coords << img.cols, img.rows;
        min_coords << -3.0, -3.0;
        max_coords << 3.0, 3.0;
        Eigen::VectorXd input_vector1 = sampleInputVector();
        Eigen::VectorXd input_vector2 = sampleInputVector();
        while (input_vector1[2] == -1.0)
        {
            input_vector1 = sampleInputVector();
        }
        while (input_vector2[2] == -1.0)
        {
            input_vector2 = sampleInputVector();
        }
        gng.initialize(input_vector1, input_vector2, min_coords, max_coords);

        // デバッグ用
        FILE *fp_nodes = fopen("GNG_result/gng_nodes.dat", "w");
        FILE *fp_edges = fopen("GNG_result/gng_edges.dat", "w");
        FILE *fp_purity = fopen("GNG_result/gng_purity.dat", "w");
        FILE *fp_hnsw = fopen("GNG_result/hnsw_edges.dat", "w");
        FILE *fp_old = fopen("GNG_result/old_nodes.dat", "w");
        if (fp_nodes == nullptr || fp_edges == nullptr || fp_purity == nullptr || fp_hnsw == nullptr || fp_old == nullptr)
        {
            std::cerr << "Error opening file for writing." << std::endl;
            return -1;
        }

        std::set<GNG::Node *> old_nodes;

        clock_t start_time = clock(); // 開始時間を記録

        for (int i = 0; i < 500000; i++)
        {
            if (i % 1000 == 0)
            {
                std::cout << "Step: " << i << " Nodes: " << gng.nodes.size() << std::endl;
                
                Eigen::ComplexEigenSolver<Eigen::MatrixXd> ces;
                Eigen::VectorXd v_y(2);
                v_y << 0.0, -1.0;
                Eigen::VectorXd v_main(2), v_sub(2);
                for (const auto node : gng.nodes)
                {
                    ces.compute(node->vcm);
                    Eigen::MatrixXcd d;
                    d = ces.eigenvalues().asDiagonal();
                    v_main << 1.0, (d(1, 1).real() - node->vcm(0, 0)) / node->vcm(1, 0);
                    v_main.normalize();
                    v_main *= sqrt(d(1, 1).real()); // 2標準偏差
                    v_sub << 1.0, (d(0, 0).real() - node->vcm(0, 0)) / node->vcm(1, 0);
                    v_sub.normalize();
                    v_sub *= sqrt(d(0, 0).real()); // 2標準偏差

                    double deg = acos(v_main.dot(v_y) / (v_y.norm() * v_main.norm())) / M_PI * 180 - 90;
                    // std::cout << d << std::endl;
                    fprintf(fp_nodes, "%f %f %f %f %f %f %f %f %f %f %f\n",
                            node->reference_vector[0],
                            node->reference_vector[1],
                            node->purity,
                            2 * sqrt(d(0, 0).real()), 
                            2 * sqrt(d(1, 1).real()), 
                            deg, 
                            v_main[0],
                            v_main[1], 
                            v_sub[0],
                            v_sub[1], 
                            (std::log(gng.hnsw.node_count + 1) - node->hnsw_node->level) * gng.hnsw.mL
                        );
                    for (const auto &[neighbor, edge] : node->neighbors)
                    {
                        fprintf(fp_hnsw, "%f %f %f %f %d\n",
                                node->reference_vector[0],
                                node->reference_vector[1],
                                neighbor->reference_vector[0],
                                neighbor->reference_vector[1],
                                std::min(static_cast<int>(std::floor((gng.hnsw.log_node_count - neighbor->hnsw_node->level) * gng.hnsw.mL)),
                                static_cast<int>(std::floor((gng.hnsw.log_node_count - node->hnsw_node->level) * gng.hnsw.mL)))
                                );
                        if (edge->weight == 0.0)
                            continue;
                        fprintf(fp_edges, "%f %f %f %f %f %f %f\n",
                                node->reference_vector[0],
                                node->reference_vector[1],
                                node->purity,
                                neighbor->reference_vector[0],
                                neighbor->reference_vector[1],
                                neighbor->purity,
                                edge->weight);
                    }
                }
                if (gng.nodes.size() >= 2 && old_nodes.empty())
                {
                    for (const auto node : gng.nodes)
                    {
                        old_nodes.insert(node);
                    }
                }
                for (const auto node : old_nodes)
                {
                    fprintf(fp_old, "%f %f\n",
                            node->reference_vector[0],
                            node->reference_vector[1]);
                }
                if (old_nodes.empty())
                {
                    fprintf(fp_old, "0\n");
                }
                fprintf(fp_nodes, "\n\n");
                fprintf(fp_edges, "\n\n");
                fprintf(fp_hnsw, "\n\n");
                fprintf(fp_old, "\n\n");
                fflush(fp_nodes);
                fflush(fp_edges);
                fflush(fp_hnsw);
                fflush(fp_old);
            }
            gng.run(sampleInputVector());

            // if (gng.step == 0)
            // {
            //     std::pair<GNG::Node *, GNG::Node *> nodes_u_f = gng.findNodesUandF();
            //     GNG::Node *max_error_node = nodes_u_f.first;
            //     double min_purity = nodes_u_f.second->purity;
            //     for (const auto node : gng.nodes)
            //     {
            //         if (node->purity < min_purity)
            //         {
            //             min_purity = node->purity;
            //         }
            //     }
                
            //     fprintf(fp_purity, "%d %d %f %f\n", gng.cycle, gng.nodes.size(), max_error_node->error_radius, min_purity);
            //     fflush(fp_purity);
            // }
        }

        fclose(fp_nodes);
        fclose(fp_edges);
        fclose(fp_purity);
        fclose(fp_hnsw);

        clock_t end_time = clock();                                                        // 終了時間を記録
        double elapsed_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC; // 秒単位に変換
        std::cout << "Total time for GNG: " << elapsed_time << " seconds" << std::endl;
        std::cout << "Levels in HNSW: " << (std::log(gng.hnsw.node_count) - gng.hnsw.entry_node->level) * gng.hnsw.mL << std::endl;
        std::cout << "Wrong counts in NSW: " << wrong_count << std::endl;
        std::cout << "Winner1 accuracy in NSW: " << static_cast<double>(winner1_count_nsw) / winner_count_bf * 100.0 << " %" << std::endl;
        std::cout << "Total accuracy in NSW: " << static_cast<double>(total_count_nsw) / winner_count_bf * 100.0 << " %" << std::endl;
    }
    return 0;
}