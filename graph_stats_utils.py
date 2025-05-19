import networkx as nx
import numpy as np
from typing import Dict, Any
from scipy.stats import entropy, wasserstein_distance

def collect_graph_stats(G: nx.Graph) -> Dict[str, Dict[str, Any]]:
    """
    Collects various statistics of a graph. Degree, shortest path, centrality,
    density, eulerian, clique, and connected components, etc.
    """
    stats = {}
    stats["max_degree"] = max(dict(G.degree()).values())
    stats["min_degree"] = min(dict(G.degree()).values())
    stats["avg_degree"] = sum(dict(G.degree()).values()) / len(dict(G.degree()).values())
    stats["shortest_path"] = nx.average_shortest_path_length(G)
    stats["avg_centrality"] = np.mean(list(nx.degree_centrality(G).values()))
    stats["density"] = nx.density(G)
    stats["eulerian"] = nx.is_eulerian(G)
    stats["clique"] = nx.graph_number_of_cliques(nx.MultiGraph(G))
    stats["connected_components"] = nx.number_connected_components(nx.MultiGraph(G))
    stats["num_triangles"] = sum(nx.triangles(nx.Graph(G)).values()) / 3
    stats["diameter"] = nx.diameter(G)
    stats["avg_clustering"] = nx.average_clustering(nx.Graph(G))
    stats["efficiency"] = nx.global_efficiency(nx.MultiGraph(G))
    stats["local_efficiency"] = nx.local_efficiency(nx.MultiGraph(G))
    stats["avg_shortest_path"] = nx.average_shortest_path_length(G)
    stats["avg_eccentricity"] = np.mean(list(nx.eccentricity(G).values()))  # max distance from a node to all other nodes
    stats["radius"] = nx.radius(G)  # min eccentricity
    stats["total_triads"] = sum(nx.triadic_census(G).values())
    stats["triad_percentage"] = nx.transitivity(nx.Graph(G))
    return stats


def get_feature_nodes(G: nx.Graph) -> Dict[str, Dict[str, Any]]:
    """
    Returns the feature nodes of a graph.
    """
    stats = {}
    stats["max_degree_node"] = max(dict(G.degree()).items(), key=lambda x: x[1])
    stats["min_degree_node"] = min(dict(G.degree()).items(), key=lambda x: x[1])
    stats["max_centrality_node"] = max(nx.degree_centrality(G).items(), key=lambda x: x[1])
    stats["min_centrality_node"] = min(nx.degree_centrality(G).items(), key=lambda x: x[1])
    stats["max_betweenness_node"] = max(nx.betweenness_centrality(G).items(), key=lambda x: x[1])
    stats["min_betweenness_node"] = min(nx.betweenness_centrality(G).items(), key=lambda x: x[1])
    stats["max_closeness_node"] = max(nx.closeness_centrality(G).items(), key=lambda x: x[1])
    stats["min_closeness_node"] = min(nx.closeness_centrality(G).items(), key=lambda x: x[1])
    stats["barycenter"] = nx.barycenter(G)  # median distance
    stats["weighted_barycenter"] = nx.barycenter(G, weight='length')  # median distance
    stats["periphery"] = nx.periphery(G)
    stats["center"] = nx.center(G)  # eccentricity = radius
    stats["pattern_stats"] = nx.triadic_census(G)

    return stats


def collect_graph_metrics(G: nx.Graph) -> Dict[str, Dict[str, Any]]:
    """
    Collects various metrics of a graph. Entropy, wasserstein distance, etc.
    """
    metrics = {}
    metrics["degree_entropy"] = entropy(list(dict(G.degree()).values()))
    metrics["degree_wasserstein"] = wasserstein_distance(list(dict(G.degree()).values()), list(dict(G.degree()).values()))
    metrics["centrality_entropy"] = entropy(list(nx.degree_centrality(G).values()))
    metrics["centrality_wasserstein"] = wasserstein_distance(list(nx.degree_centrality(G).values()), list(nx.degree_centrality(G).values()))
    metrics["betweenness_entropy"] = entropy(list(nx.betweenness_centrality(G).values()))
    metrics["betweenness_wasserstein"] = wasserstein_distance(list(nx.betweenness_centrality(G).values()), list(nx.betweenness_centrality(G).values()))
    metrics["closeness_entropy"] = entropy(list(nx.closeness_centrality(G).values()))
    metrics["closeness_wasserstein"] = wasserstein_distance(list(nx.closeness_centrality(G).values()), list(nx.closeness_centrality(G).values()))
    return metrics


def get_degree_distribution(G: nx.Graph) -> Dict[int, int]:
    """
    Returns the degree distribution of a graph.
    """
    degree_dist = {}
    for node, degree in G.degree():
        if degree in degree_dist:
            degree_dist[degree] += 1
        else:
            degree_dist[degree] = 1
    return degree_dist


if __name__ == "__main__":
    import pickle
    saved_data = pickle.load(
        open("/home/srg/projects/urbanity/saved_data/Chiang Mai.pkl", "rb"))
    network, intersections, streets = saved_data["network"], saved_data[
        "intersection"], saved_data["street"]
    graph = network
    stats = collect_graph_stats(graph)
    print(stats)
    degree_dist = get_degree_distribution(graph)
    print(degree_dist)
