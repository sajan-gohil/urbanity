import community_utils
import graph_stats_utils
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from typing import Dict, Any
from scipy.stats import entropy, wasserstein_distance
from collections import Counter

def plot_degree_distribution(G: nx.Graph):
    """
    Plots the degree distribution of a graph.
    """
    degree_dist = graph_stats_utils.get_degree_distribution(G)
    degree_dist = dict(sorted(degree_dist.items(), key=lambda x: x[0]))
    degree_dist = dict(Counter(degree_dist).most_common())
    plt.figure(figsize=(10, 6))
    plt.bar(degree_dist.keys(), degree_dist.values())
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title("Degree Distribution")
    plt.show()


def plot_community_distribution(G: nx.Graph):
    """
    Plots the community distribution of a graph.
    """
    modularity, num_communities, partition = community_utils.detect_communities(G)
    community_dist = dict(Counter(partition[1:].values()))
    community_dist = dict(sorted(community_dist.items(), key=lambda x: x[0]))
    plt.figure(figsize=(10, 6))
    plt.bar(community_dist.keys(), community_dist.values())
    plt.xlabel("Community")
    plt.ylabel("Frequency")
    plt.title("Community Distribution")
    plt.show()


def plot_feature_nodes(G: nx.Graph):
    """
    Plots the feature nodes of a graph.
    """
    feature_nodes = graph_stats_utils.get_feature_nodes(G)
    feature_nodes = dict(sorted(feature_nodes.items(), key=lambda x: x[0]))
    plt.figure(figsize=(10, 6))
    plt.bar(feature_nodes.keys(), feature_nodes.values())
    plt.xlabel("Feature Node")
    plt.ylabel("Value")
    plt.title("Feature Nodes")
    plt.show()


def plot_graph_stats(G: nx.Graph):
    """
    Plots the statistics of a graph.
    """
    stats = graph_stats_utils.collect_graph_stats(G)
    stats = dict(sorted(stats.items(), key=lambda x: x[0]))
    plt.figure(figsize=(10, 6))
    plt.bar(stats.keys(), stats.values())
    plt.xlabel("Statistic")
    plt.ylabel("Value")
    plt.title("Graph Statistics")
    plt.show()

    # combined plot for all stats
    df = pd.DataFrame(stats.items(), columns=["Statistic", "Value"])
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Statistic", y="Value", data=df)
    plt.title("Graph Statistics")
    plt.show()

    # correlation between stats
    df = pd.DataFrame(stats)
    corr = df.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True)
    plt.title("Correlation between Graph Statistics")
    plt.show()


def plot_graph_metrics(G: nx.Graph):
    """
    Plots the metrics of a graph.
    """
    metrics = graph_stats_utils.collect_graph_metrics(G)
    metrics = dict(sorted(metrics.items(), key=lambda x: x[0]))
    plt.figure(figsize=(10, 6))
    plt.bar(metrics.keys(), metrics.values())
    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.title("Graph Metrics")
    plt.show()


def plot_degree_centrality(G: nx.Graph):
    """
    Plots the degree centrality of a graph.
    """
    degree_centrality = nx.degree_centrality(G)
    degree_centrality = dict(sorted(degree_centrality.items(), key=lambda x: x[0]))
    plt.figure(figsize=(10, 6))
    plt.bar(degree_centrality.keys(), degree_centrality.values())
    plt.xlabel("Node")
    plt.ylabel("Degree Centrality")
    plt.title("Degree Centrality")
    plt.show()


def aggregate_stats_for_subzones(graph_dict: dict[tuple[str, str], nx.Graph]) -> Dict[str, Any]:
    """
    Aggregates the statistics of subzones.
    """
    stats = {}
    for subzone, G in graph_dict.items():
        # print("processing: ", subzone, len(G.nodes))
        stats[subzone] = graph_stats_utils.collect_graph_stats(G)
        # print(subzone, stats, len(G.nodes))
    df = pd.DataFrame(stats).T
    df = df.reset_index()
    df = df.rename(columns={"index": "Subzone"})
    return df


def aggregate_metrics_for_subzones(graph_dict: dict[tuple[str, str], nx.Graph]) -> Dict[str, Any]:
    """
    Aggregates the metrics of subzones.
    """
    metrics = {}
    for subzone, G in graph_dict.items():
        metrics[subzone] = graph_stats_utils.collect_graph_metrics(G)
    df = pd.DataFrame(metrics).T
    df = df.reset_index()
    df = df.rename(columns={"index": "Subzone"})
    return df

def plot_subzone_correlation(df: pd.DataFrame):
    """
    Plots the correlation between subzones.
    """
    corr = df.corr()
    color = plt.get_cmap('coolwarm')   # default color
    color.set_bad('lightgray')
    sns.heatmap(corr, annot=True, cmap=color)
    plt.title("Correlation between Subzones")
    plt.show()


def analyze_supergraph(G, city_name):
    # Detect communities using greedy modularity optimization
    communities = list(nx.community.greedy_modularity_communities(G))

    # Assign community labels to nodes
    partition = {
        node: i
        for i, comm in enumerate(communities)
        for node in comm
    }
    for node in G.nodes():
        G.nodes[node]['cluster'] = partition[node]

    # Create a super-graph where nodes represent clusters
    super_graph = nx.Graph()

    # Add nodes (clusters) to the super-graph
    for node in G.nodes():
        cluster_id = G.nodes[node]['cluster']
        super_graph.add_node(cluster_id)

    # Add edges between clusters with aggregated weights
    for u, v in G.edges():
        cluster_u = G.nodes[u]['cluster']
        cluster_v = G.nodes[v]['cluster']
        if cluster_u != cluster_v:
            if super_graph.has_edge(cluster_u, cluster_v):
                super_graph[cluster_u][cluster_v]['weight'] += 1
            else:
                super_graph.add_edge(cluster_u, cluster_v, weight=1)

    # Print graph statistics
    print("Original Graph Nodes:", G.number_of_nodes())
    print("Original Graph Edges:", G.number_of_edges())
    print("Super-Graph Nodes:", super_graph.number_of_nodes())
    print("Super-Graph Edges:", super_graph.number_of_edges())

    # Visualize the super-graph
    plt.figure(figsize=(12, 12))
    nx.draw_kamada_kawai(super_graph,
                         with_labels=True,
                         node_color='lightblue',
                         edge_color='gray')
    plt.title("Super-Graph")
    plt.show()

    # Function to compute the sum of edge weights for a given node in the super-graph
    def sum_weights(node):
        return sum(super_graph[node][neighbor]['weight']
                   for neighbor in super_graph[node] if neighbor != node)

    # Compute node weights
    node_weights = [sum_weights(node) for node in super_graph.nodes]

    print("\n")
    # Plot histogram of node weights
    # Dummy plot (invisible, just for the legend)
    # plt.plot('',label=city_name)
    # # Use legend to display the text box
    # plt.legend(loc="upper right", frameon=True, edgecolor='black', fontsize=12)

    plt.hist(node_weights, bins=12, edgecolor='black')
    plt.title(f"Connections Between Major Zones. {city_name}")
    plt.xlabel("Degree of Connectivity")
    plt.ylabel("Number of Zones")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import os
    graph_dict = {}
    for file in os.listdir("saved_data"):
        if len(graph_dict) >= 1:
            break
        if file.endswith(".pkl"):
            with open(f"saved_data/{file}", "rb") as f:
                city_subzone = file.split('_')[0]# + '_' + file.split('_')[1]
                network = pickle.load(f)['network']
                if network is None or len(network.nodes) > 1000:
                    continue
                graph_dict[city_subzone] = network
                print(len(network.nodes))
    print(len(graph_dict))
    # df = aggregate_stats_for_subzones(graph_dict)
    # print(df)
    # df.to_csv("subzone_stats.csv", index=False)
    # plot_subzone_correlation(df.drop(columns=["Subzone"]))
    # df = aggregate_metrics_for_subzones(graph_dict)
    # plot_subzone_correlation(df)


    for subzone, G in graph_dict.items():
        print(subzone)
        plot_degree_distribution(G)
        plot_community_distribution(G)
        plot_feature_nodes(G)
        plot_graph_stats(G)
        plot_graph_metrics(G)
        plot_degree_centrality(G)
        break
