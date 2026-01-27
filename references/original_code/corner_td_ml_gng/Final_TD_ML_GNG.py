#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:37:02 2024

@author: kubota
"""

import numpy as np
import cv2
import math
import random
import time

ERROR_INDEX = -4
W2_INDEX = -3
ACTIVATION_INDEX = -2
PARANT_INDEX = -1


EDGE_AGE = -1

class GNG:
  
    def __init__(self, feature_number = 2, maxNodeLength = 68, L1 = 0.5, L2 = 0.01, 
                 newNodeFreq = 50, maxAge = 25, newNodeFactor = 0.5):
        
        self.child_layer = None
        self.parent_layer = None
        
        # 特徴量の次元数を保存
        self.feature_number = feature_number
        
        # W配列の初期化（feature_number + 4の次元で初期化）
        self.W = np.empty((0, feature_number + 4))   
        self.c = np.empty((0, 3), dtype=int)
        
        self.M = maxNodeLength
        self.alpha = L1
        self.beta = L2
        self.gamma = newNodeFreq        
        self.theta = maxAge
        self.rho = newNodeFactor
        
        self.eps = 0.000001
        self.inputCount = 0
        self.removeCount = 0
        
    
   
    def connectTwoNodes(self, s1, s2):
        c1 = np.logical_and(self.c[:,0] == s1, self.c[:,1] == s2)
        c2 = np.logical_and(self.c[:,0] == s2, self.c[:,1] == s1)
        c = np.logical_or(c1,c2)       
        if True in c:               
            self.c[c==True, EDGE_AGE] = 0
        else:
            self.c = np.vstack((self.c,[s1,s2,0]))
    
    
    def findNearestNodes(self, x, subNetworkID):
        
   
        
        x2 = np.sum(np.square(x))
        
        dist = self.W[subNetworkID,W2_INDEX] - 2 * np.matmul(x, self.W[subNetworkID,:ERROR_INDEX].T)
        
        
        s1 = np.argmin(dist)    
        temp = dist[s1] + x2
        if temp > 0:
            d_s1 = np.sqrt(temp)
        else:
            d_s1 = 0
        

        
        dist[s1] = 99999
        s2 = np.argmin(dist)
        
       
       
        
        
        return subNetworkID[s1], subNetworkID[s2], d_s1
    
    
    def getConnectedNodes(self, index):
        connectedNodes = np.unique(np.concatenate((self.c[self.c[:,0] == index,1], self.c[self.c[:,1] == index,0])))
        return connectedNodes
    
    def removeOldEdges(self):
        
        # delete edge if more then age.
        nodeToCheck = self.c[self.c[:,EDGE_AGE] >= self.theta][:,:2]
        self.c = self.c[self.c[:,EDGE_AGE] < self.theta]        
        nodeToCheck = np.unique(nodeToCheck)
        return nodeToCheck
    
    
    def checkRemoveIsolated(self, nodeToCheck):
        
        #check node required to delete
        finalDelete = []
        for v in nodeToCheck:            
            if v not in self.c[:,:2]:               
                finalDelete.append(v)
        
        
        return finalDelete
    
    
        
      
    def deleteNodes(self, nodeToDelete):
        
        if len(nodeToDelete) > 1:
            nodeToDelete.sort(reverse=True)   
            

        for v in nodeToDelete:   
    
            if self.child_layer is not None:
                totalChildren = np.sum(self.child_layer.W[:,PARANT_INDEX] == v)
                if totalChildren >= 1:
                    # print("Has Children", np.sum(self.child_layer.W[:,PARANT_INDEX] == v))                    
                    continue
              
                    
            if v in self.winner_idx:
                continue
            
            parent_id = int(self.W[v,PARANT_INDEX])
            
            # Remove connected Edges
            self.c = self.c[np.logical_not(np.logical_or(self.c[:,0] == v, self.c[:,1] == v))]
            # Reduce the connecting index
            self.c[self.c[:,0] > v,0] -= 1
            self.c[self.c[:,1] > v,1] -= 1           
            
            self.winner_idx[self.winner_idx > v] -= 1
            #Remove nodes from network
            self.W = np.delete(self.W, v, axis=0)
            
                        
            if self.parent_layer is not None:
                if parent_id not in self.W[:,PARANT_INDEX]:
                    self.parent_layer.deleteNodes([parent_id])  
            
            if self.child_layer is not None:  
                self.child_layer.W[self.child_layer.W[:,PARANT_INDEX] > v,PARANT_INDEX] -= 1
             
                    
            
    def getConnectedDistance(self, s1,s2):
        c1 = np.logical_or(self.c[:,0] == s1, self.c[:,1] == s1)
        c2 = np.logical_or(self.c[:,0] == s2, self.c[:,1] == s2)
        cond = np.logical_or(c1,c2)
        
        dist = np.linalg.norm(self.W[self.c[cond,0],:ERROR_INDEX] - self.W[self.c[cond,1],:ERROR_INDEX], axis=1)
        return dist
    
    
    def checkRemoveInactive(self):
        idx = np.argwhere(self.W[:,ACTIVATION_INDEX] == 0).T[0]
       
        nodesToDelete = []
        for s1 in idx:
            if random.random()<0.1:                
                nodesToDelete.append(s1)
         

        self.deleteNodes(nodesToDelete)
        
        self.W[:,ACTIVATION_INDEX] *= 0
        
        
        # Remove Isolated Nodes
        isolatedNodes = self.checkRemoveIsolated(np.arange(len(self.W)))
        self.deleteNodes(isolatedNodes)
    
    
    def getSubnetwork(self, parent_idx):
   
        subW = self.W[:, PARANT_INDEX] == parent_idx[0]
        for i in range(1, len(parent_idx)):
            subW = np.logical_or(subW, self.W[:, PARANT_INDEX] == parent_idx[i])
        
        subW_id = np.argwhere(subW).T[0]
        return subW_id
        
                
                
    
    def pushData(self, x, parent_idx ):
        if len(self.W) < 2:
            # 新しいノードを作成する際の次元数を明示的に指定
            new_w = np.zeros(self.feature_number + 4)  # 特徴量 + 4つの追加情報
            new_w[:self.feature_number] = x  # 特徴量をコピー
            new_w[W2_INDEX] = np.sum(np.square(x))  # W2の計算
            new_w[ACTIVATION_INDEX] = 1  # アクティベーション
            new_w[PARANT_INDEX] = parent_idx[0]  # 親ノードのインデックス
            
            self.W = np.vstack((self.W, new_w))
            return [len(self.W) - 1]
        
        
        subW_id = self.getSubnetwork(parent_idx)   
        if len(subW_id) <= 0:
            # print("Empty sub network")
            q3 = self.insertNewNode(x, parent_idx[0])
            return [q3]
        
        
        s1, s2, d_s1 =  self.findNearestNodes(x, subW_id)
        self.winner_idx = np.asarray([s1,s2])
        
        # # check connection, if connected then set age to 0, else add new age
        self.connectTwoNodes(s1, s2)
        
        self.W[s1,ACTIVATION_INDEX] += 1
        
        
        if len(self.W) < self.M:            
            dist = self.getConnectedDistance(s1,s2)            
            z = d_s1  - np.mean(dist)
            p = max(0,math.tanh(z /( np.max(dist) + self.eps)))     
           
            if p > random.random():                      
              
                
                s2 = self.addIfSilent(x, s1)
                return [-1]
        
        #increase age
        self.c[np.logical_or(self.c[:,0] == s1, self.c[:,1] == s1), EDGE_AGE] += 1               
        
        
        
        # increase winner node error
        self.W[s1,ERROR_INDEX] += d_s1 * self.alpha
        
        # move the winner node        
        self.W[s1,:ERROR_INDEX] += (x - self.W[s1,:ERROR_INDEX]) * self.alpha
        self.W[s1,W2_INDEX] = np.sum(np.square(self.W[s1,:ERROR_INDEX]))
        
        
        # find the neighbor nodes
        connectedNodes = self.getConnectedNodes(s1)
        
        # move neighbor nodqe            
        self.W[connectedNodes,:ERROR_INDEX] += (x - self.W[connectedNodes,:ERROR_INDEX]) * self.beta
        self.W[connectedNodes, W2_INDEX] = np.sum(np.square(self.W[connectedNodes,:ERROR_INDEX]),axis=1)
        
        
        if self.parent_layer is not None:
            # self.W[s1,PARANT_INDEX] = self.parent_layer.getNearestNode(self.W[s1,:ERROR_INDEX])
            temp = np.append(connectedNodes, s1)
            self.updateParentLink(temp)
        
        # Remove old edges
        #nodes to check is it already isolated, instead of search through the entire network to remove isolated nodes
        nodesToCheck = self.removeOldEdges()    
        
        # Remove Isolated Nodes
        isolatedNodes = self.checkRemoveIsolated(nodesToCheck)
        self.deleteNodes(isolatedNodes)
        
        
        # For every input interval
        self.inputCount += 1
        if self.inputCount >= self.gamma:
  
            self.inputCount = 0
            
            
            g = self.calculateTotalNodesAdd()
            #add new node
            for _ in range(g):
                self.addNewNode()
                
     

        self.removeCount += 1
        if self.removeCount > len(self.W) *5: #/ math.tanh(len(self.W)/400) :    
            self.removeCount = 0
            self.checkRemoveInactive()
       
        
        return self.winner_idx
        
    
    def calculateTotalNodesAdd(self):
        data = self.W[:,ERROR_INDEX]
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
       
        iqr = q3 - q1
        threshold = 1.5 * iqr
        outliers = np.where((data > q3 + threshold))

        return len(outliers) + 1
    
    def addIfSilent(self, x, s1):
        q3 = len(self.W)
        
        new_w = np.append(x, [0, 0,1,-1])
        new_w[W2_INDEX] = np.sum(np.square(x))
        if self.parent_layer is not None:
            new_w[PARANT_INDEX] = self.parent_layer.getNearestNode(new_w[:ERROR_INDEX])
            
        self.W = np.vstack((self.W,new_w))     
        self.c = np.vstack((self.c,[s1,q3,0]))
        self.W[s1,ERROR_INDEX] *= 0.5
        
        
        if self.child_layer is not None:
            self.child_layer.insertNewNode(new_w[:ERROR_INDEX], q3)
        return q3
        
    
    def getNearestNode(self, x):
        
        
        x2 = np.sum(np.square(x))
                
        dist = self.W[:,W2_INDEX] - 2 * np.matmul(x, self.W[:,:ERROR_INDEX].T)
        return np.argmin(dist)
        
    
    def addNewNode(self):
        
        if len(self.W) >= self.M:
            return
        
        # get the maximum error
        q1 = np.argmax(self.W[:,ERROR_INDEX])        
        # get the connected nodes
        connectedNodes = self.getConnectedNodes(q1)
        
        if len(connectedNodes) == 0:            
            print("No next highest error node")            
            return
       
        # get the maximum error of neighbors
        q2 = connectedNodes[np.argmax(self.W[connectedNodes, ERROR_INDEX])]
        
        error_all = self.W[q1, ERROR_INDEX] + self.W[q2, ERROR_INDEX] + self.eps

       
        # insert new node between q1 and q2 based on error rate
        q3 = len(self.W)
        r1 = self.W[q1, ERROR_INDEX]  / error_all              
        r2 = 1 - r1
        
       
        new_w = (self.W[q1] * r1 + self.W[q2] * r2)

        new_w[ACTIVATION_INDEX] = 1
        new_w[PARANT_INDEX] = -1
        new_w[W2_INDEX] = np.sum(np.square(new_w[:ERROR_INDEX]))
        
        if self.parent_layer is not None:
            new_w[PARANT_INDEX] = self.parent_layer.getNearestNode(new_w[:ERROR_INDEX])
        
            
        self.W = np.vstack((self.W,new_w))     
        self.W[q1, ERROR_INDEX] -= self.W[q1, ERROR_INDEX] * self.rho
        self.W[q2, ERROR_INDEX] -= self.W[q2, ERROR_INDEX] * self.rho
        self.W[q3, ERROR_INDEX] = (self.W[q1, ERROR_INDEX] + self.W[q2, ERROR_INDEX]) * 0.5
        
        
    
        #remove the original edge
        self.c = self.c[ np.logical_not(np.logical_and(self.c[:,0] == q1, self.c[:,1] == q2))]
        self.c = self.c[ np.logical_not(np.logical_and(self.c[:,0] == q2, self.c[:,1] == q1))]
        #add the edge
        self.c = np.vstack((self.c,[q1,q3,0]))
        self.c = np.vstack((self.c,[q2,q3,0]))     

                
        if self.child_layer is not None:
            self.child_layer.insertNewNode(new_w[:ERROR_INDEX], q3)
        
        
    def insertNewNode(self, x, parent_index): 
        q3 = len(self.W)
        
        # 新しいノードの作成時に次元数を明示的に指定
        new_w = np.zeros(self.feature_number + 4)
        new_w[:self.feature_number] = x  # 特徴量をコピー
        new_w[ERROR_INDEX] = 0
        new_w[W2_INDEX] = np.sum(np.square(x))
        new_w[ACTIVATION_INDEX] = 1
        new_w[PARANT_INDEX] = parent_index
        
        # 最も近いノードを見つける
        if len(self.W) > 0:
            s1 = self.getNearestNode(x)
        else:
            s1 = 0
        
        self.W = np.vstack((self.W, new_w))     
        self.c = np.vstack((self.c, [s1, q3, 0]))
        
        if s1 < len(self.W):  # インデックスの範囲チェック
            self.W[s1, ERROR_INDEX] *= 0.5
        
        if self.child_layer is not None:
            self.child_layer.insertNewNode(new_w[:self.feature_number], q3)
        
        return q3
    
    def updateParentLink(self, nodes_idx):
        x2 = self.parent_layer.W[:,W2_INDEX]
        y2 = self.W[nodes_idx,W2_INDEX]     
        dot_product = 2 * np.matmul(self.parent_layer.W[:,:ERROR_INDEX], self.W[nodes_idx,:ERROR_INDEX].T)   
        dist = np.expand_dims(x2, axis=1) + y2 - dot_product
        self.W[nodes_idx,PARANT_INDEX] = np.argmin(dist.T,axis=1)
        
        
    def hirahicalUpdate(self):       
        x2 = self.parent_layer.W[:,W2_INDEX]
        y2 = self.W[:,W2_INDEX]     
        dot_product = 2 * np.matmul(self.parent_layer.W[:,:ERROR_INDEX], self.W[:,:ERROR_INDEX].T)         
        dist = np.expand_dims(x2, axis=1) + y2 - dot_product
        self.W[:,PARANT_INDEX] = np.argmin(dist.T,axis=1)
        
        return dist
    
    
    def addParentNode(self, x, s1):
        q3 = len(self.W)
        
        new_w = np.append(x, [0, 0,1,-1])
        new_w[W2_INDEX] = np.sum(np.square(x))
        if self.parent_layer is not None:
            new_w[PARANT_INDEX] = self.parent_layer.getNearestNode(new_w[:ERROR_INDEX])
            
        self.W = np.vstack((self.W,new_w))     
        self.c = np.vstack((self.c,[s1,q3,0]))
        self.W[s1,ERROR_INDEX] *= 0.5
        
    
    def hirahicalSplit(self):           
        dd = np.linalg.norm(self.child_layer.W[self.child_layer.c[:,0],:ERROR_INDEX] - self.child_layer.W[self.child_layer.c[:,1],:ERROR_INDEX], axis=1)
        radius = np.mean(dd)
        
        
        hasAddNewNode = False
        for i in range(len(self.W)):
      
            particles_idx = np.argwhere(self.child_layer.W[:,PARANT_INDEX] == i).T[0]
            if len(particles_idx) <= 3:
                continue
            samplePoint = self.W[i,:ERROR_INDEX]
            particles =  self.child_layer.W[particles_idx,:ERROR_INDEX]
            dist = np.linalg.norm(samplePoint - particles, axis=1)
            
        
            density = np.exp(-dist**2/(2*radius**2))
            
            
            if np.mean(density) < 0.5:
                x = particles[np.argmax(dist)]
                self.addParentNode(x, i)
                hasAddNewNode = True
            
        if hasAddNewNode:            
            self.child_layer.hirahicalUpdate()
            return True
        
        return False
        
    def getTotalError(self, X):
        W = self.W
        totalError = 0
        for x in X:
            diff = x - W[:,:ERROR_INDEX] 
            dist = np.linalg.norm(diff, axis=1)    
            
            idx = np.argsort(dist)        
            s1 = idx[0]
            totalError += dist[s1]
        return totalError / len(X)



class TDMLGNG:
    def __init__(self, featureNumber, maxLayerNodes):
        self.spllitingCount = 0        
        self.layers = []
        self.epoch_count = 0
        
        assert len(maxLayerNodes) >= 2, "At least two layers"
        for i in range(len(maxLayerNodes) - 1):
            assert maxLayerNodes[i] < maxLayerNodes[i + 1], "Upper layer must be lower than lower layer"

        #Maximum Nodes
        L = len(maxLayerNodes)
        #Winner node learning rate
        L1 = 0.5
        #Neighbor nodes learning rate
        L2 = 0.01
        #Create New Node Frequency
        newNodeFrequency = 50
        
        edgeMaxAge = 25
                
        for l, M in enumerate(maxLayerNodes):
            alpha = L1 / math.pow(10, L - l -1)
            beta = L2 / math.pow(10, L - l -1)
            gamma = newNodeFrequency * (L-l)  
            theta = edgeMaxAge * (L-l)        
            # 上の層ほど更新頻度を下げ、エッジの寿命を長くする
            # alpha = L1 / math.pow(10, l)  # 上の層ほど学習率を小さく
            # beta = L2 / math.pow(10, l)   # 上の層ほど学習率を小さく
            # gamma = newNodeFrequency * (l + 1)  # 下の層ほど頻繁に新しいノードを追加
            # theta = edgeMaxAge * (l + 1)        # 下の層ほどエッジの寿命を短く
            
            print(f"Layer {l} parameters:")
            print(f"  alpha: {alpha}")
            print(f"  beta: {beta}")
            print(f"  gamma: {gamma}")
            print(f"  theta: {theta}")
            gng = GNG(featureNumber, M, alpha, beta, gamma, theta)
            
            if len(self.layers) > 0:
                gng.parent_layer = self.layers[-1]
                self.layers[-1].child_layer = gng
            
            self.layers.append(gng)
        
        print("Initialized TDMLGNG with following configuration:")
        for l, M in enumerate(maxLayerNodes):
            print(f"Layer {l}: Max nodes = {M}")
            
    
    def log_network_stats(self, epoch=None):
        """ネットワークの現在の状態をログ出力"""
        if epoch is not None:
            print(f"\nEpoch {epoch} Network Statistics:")
        else:
            print("\nCurrent Network Statistics:")
            
        total_nodes = 0
        for l, gng in enumerate(self.layers):
            nodes = len(gng.W)
            edges = len(gng.c)
            max_error = np.max(gng.W[:, ERROR_INDEX]) if nodes > 0 else 0
            avg_error = np.mean(gng.W[:, ERROR_INDEX]) if nodes > 0 else 0
            
            print(f"Layer {l}:")
            print(f"  Nodes: {nodes}")
            print(f"  Edges: {edges}")
            print(f"  Max Error: {max_error:.4f}")
            print(f"  Avg Error: {avg_error:.4f}")
            total_nodes += nodes
        
        print(f"Total nodes across all layers: {total_nodes}")
        print("-" * 50)
    
    def pushData(self, x, log_frequency=None):
        parent_idx = [-1]
        for gng in self.layers:
            parent_idx = gng.pushData(x, parent_idx)  
            if parent_idx[0] == -1:                    
                # no need to continue this data
                return 
            gng.winner_idx = np.asarray([])
        
        # ログ出力の頻度制御
        # if log_frequency is not None and self.spllitingCount % log_frequency == 0:
        #     self.log_network_stats()
            
        self.spllitingCount += 1
        for l in range(len(self.layers)-1):
            if self.spllitingCount % self.layers[l].gamma == 0:
                if self.layers[l].hirahicalSplit():
                    print(f"Hierarchical split occurred in layer {l}")
    
    def displayGraph(self, windowName, data, nodeList, edgeList, layer_index):
        padding = 2
        img_size = 512
        minX = data[:,0].min() 
        maxX = data[:,0].max()
        
        minY = data[:,1].min() 
        maxY = data[:,1].max() 
        
        lenX = maxX - minX + padding
        lenY = maxY - minY + padding
        
        scaleX = img_size/lenX
        scaleY = img_size/lenY
        
        # 各層の色を定義（BGR形式）
        layer_colors = [
            [255, 0, 0],    # 青
            [0, 255, 0],    # 緑
            [0, 0, 255],    # 赤
            [255, 255, 0],  # シアン
            [255, 0, 255],  # マゼンタ
            [0, 255, 255],  # 黄色
        ]
        
        # 層のインデックスに基づいて色を選択
        current_color = layer_colors[layer_index % len(layer_colors)]
        edge_color = [c // 2 for c in current_color]  # エッジは少し暗めの色
        
        # 背景画像の作成
        image = np.ones((img_size,img_size,3), dtype=np.uint8) * 255
        
        # データポイントの表示（グレー）
        for x,y in data:
            pos = (int((x-minX+padding//2)*scaleX), int((y-minY+padding//2)*scaleY))
            image = cv2.circle(image, pos, 4, [200,200,200], -1) 
            
        # エッジの描画
        W = nodeList
        E = edgeList      
        for e in E:
            x1, y1 = W[e[0]][:2]
            x2, y2 = W[e[1]][:2]
            start_point = (int((x1-minX+padding//2)*scaleX), int((y1-minY+padding//2)*scaleY))
            end_point = (int((x2-minX+padding//2)*scaleX), int((y2-minY+padding//2)*scaleY))
            image = cv2.line(image, start_point, end_point, edge_color, 2)     
        
        # ノードの描画
        for i, d in enumerate(W):        
            pos = (int((d[0]-minX+padding//2)*scaleX), int((d[1]-minY+padding//2)*scaleY))
            image = cv2.circle(image, pos, 7, current_color, -1)
            # ノード番号の表示（オプション）
            cv2.putText(image, str(i), (pos[0]+10, pos[1]+10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_color, 1)
        
        # ウィンドウのタイトルに層の情報を追加
        window_title = f"{windowName} (Layer {layer_index})"
        cv2.imshow(window_title, image)  
        
        return image
    
    def display(self, data):
        self.log_network_stats()  # 表示前に現在の状態をログ出力
        for l, gng in enumerate(self.layers):
            img = self.displayGraph("layer", data, gng.W, gng.c, l)        
        cv2.waitKey(2)
     
# メインの学習ループを修正
def train_tdmlgng(data, epochs=10, log_frequency=1000):
    tdmlgng = TDMLGNG(2, [5,333,1000])
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        print(f"\nStarting Epoch {epoch+1}/{epochs}")
        
        np.random.shuffle(data)
        for i, x in enumerate(data):
            tdmlgng.pushData(x, log_frequency)
            
            # 進捗表示
            if (i + 1) % (len(data) // 10) == 0:
                progress = (i + 1) / len(data) * 100
                # print(f"Progress: {progress:.1f}% of epoch {epoch+1}")
        
        # エポック終了時の統計
        # tdmlgng.log_network_stats(epoch+1)
        tdmlgng.display(data)
    
    return tdmlgng            

# 使用例
data = []
f = open("dataset/D31.txt", "r")
for x in f:
    data.append(np.asanyarray(x.split(), dtype=np.float32))
data = np.vstack(data)    
data = data[:,:2]

tdmlgng = train_tdmlgng(data, epochs=100, log_frequency=1000)
cv2.waitKey(0)



           