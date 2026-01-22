/**
 * @file test_gng.cpp
 * @brief Tests for GNG implementation
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <random>

#include "../gng.hpp"

class GNGTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate random 2D data
        std::random_device rd;
        std::mt19937 gen(42);  // Fixed seed for reproducibility
        std::uniform_real_distribution<> dis(0.0, 1.0);

        data_2d.resize(500, 2);
        for (int i = 0; i < data_2d.rows(); ++i) {
            data_2d(i, 0) = dis(gen);
            data_2d(i, 1) = dis(gen);
        }

        // Generate random 3D data
        data_3d.resize(500, 3);
        for (int i = 0; i < data_3d.rows(); ++i) {
            data_3d(i, 0) = dis(gen);
            data_3d(i, 1) = dis(gen);
            data_3d(i, 2) = dis(gen);
        }
    }

    Eigen::MatrixXd data_2d;
    Eigen::MatrixXd data_3d;
};

TEST_F(GNGTest, Initialization) {
    gng::GNG<> model;
    EXPECT_EQ(model.n_nodes(), 0);
    EXPECT_EQ(model.n_edges(), 0);
}

TEST_F(GNGTest, Fit2D) {
    gng::GNG<> model(50, 0.2, 0.006, 0.5, 0.0005, 50, 20);
    model.fit(data_2d, 5);

    EXPECT_GT(model.n_nodes(), 2);
    EXPECT_LE(model.n_nodes(), 20);
    EXPECT_GT(model.n_edges(), 0);
}

TEST_F(GNGTest, Fit3D) {
    gng::GNG<> model(50, 0.2, 0.006, 0.5, 0.0005, 50);
    model.fit(data_3d, 3);

    EXPECT_GT(model.n_nodes(), 2);
    auto nodes = model.get_nodes();
    EXPECT_EQ(nodes.cols(), 3);
}

TEST_F(GNGTest, GetGraph) {
    gng::GNG<> model(30);
    model.fit(data_2d.topRows(200), 3);

    auto nodes = model.get_nodes();
    auto edges = model.get_edges();

    EXPECT_EQ(nodes.rows(), model.n_nodes());
    EXPECT_EQ(static_cast<int>(edges.size()), model.n_edges());
}

TEST_F(GNGTest, ClusterData) {
    // Create 3 clusters
    std::random_device rd;
    std::mt19937 gen(42);
    std::normal_distribution<> dis(0.0, 0.5);

    Eigen::MatrixXd clusters(300, 2);
    for (int i = 0; i < 100; ++i) {
        clusters(i, 0) = dis(gen) + 0.0;
        clusters(i, 1) = dis(gen) + 0.0;
    }
    for (int i = 100; i < 200; ++i) {
        clusters(i, 0) = dis(gen) + 5.0;
        clusters(i, 1) = dis(gen) + 0.0;
    }
    for (int i = 200; i < 300; ++i) {
        clusters(i, 0) = dis(gen) + 2.5;
        clusters(i, 1) = dis(gen) + 4.0;
    }

    gng::GNG<> model(50, 0.2, 0.006, 0.5, 0.0005, 50, 30);
    model.fit(clusters, 10);

    EXPECT_GT(model.n_nodes(), 5);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
