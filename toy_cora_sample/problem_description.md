# Problem Description: Node Classification on the Cora Dataset

This project addresses a node classification task on the Cora citation network dataset.

**Dataset:**

*   The Cora dataset consists of 2708 scientific publications classified into one of seven classes.
*   The citation network consists of 5429 links (edges) where each link represents a citation from one paper to another.
*   Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from a dictionary. The dictionary consists of 1433 unique words.

**Task:**

*   The task is **semi-supervised node classification**.
*   Given the network structure (citations) and the features (word vectors) of a small subset of labeled nodes (papers with known subjects), the goal is to predict the class (subject) of the unlabeled nodes.

**Goal:**

The primary objective is to train a Graph Neural Network (GNN) model, specifically a Graph Convolutional Network (GCN), that can accurately classify the papers into their respective subjects based on both their content features and the citation graph structure. 