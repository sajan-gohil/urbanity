import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import networkx as nx
import pandas as pd
import geopandas as gpd
from matplotlib import cm
import time
import osmnx
from matplotlib.font_manager import FontProperties
# import infomap
import leidenalg
import igraph as ig

def detect_communities(
        graph=None,
        algorithm='louvain',
        resolution=1.0,
        weight='length',
        dual=False,
        k=10,
        threshold=0.1,
        normalize=True,
        return_nodes=True,
        return_edges=True):
    """Function to detect communities in either primal or dual urban network.
    
    Args:
        graph (nx.Graph, optional): The network graph to analyze. If None, uses the previously computed network.
        algorithm (str): Community detection algorithm to use. Options: 'louvain', 'leiden', 
                         'label_propagation', 'girvan_newman', 'asyn_fluidc', 'greedy_modularity'.
                         Defaults to 'louvain'.
        resolution (float): Resolution parameter for community detection. Higher values result in more communities.
                           Only applicable for Louvain and Leiden algorithms. Defaults to 1.0.
        weight (str): Edge attribute to use as weight. Defaults to 'length'.
        dual (bool): If True, uses the dual representation of the network. Defaults to False.
        k (int): Number of communities for spectral clustering. Only used if algorithm is 'spectral'. Defaults to 10. (Not implemented yet)
        threshold (float): Threshold for edge removal in Girvan-Newman. Only used if algorithm is 'girvan_newman'.
                          Defaults to 0.1.
        normalize (bool): If True, normalizes edge weights before community detection. Defaults to True.
        return_nodes (bool): If True, returns nodes GeoDataFrame with community assignments. Defaults to True.
        return_edges (bool): If True, returns edges GeoDataFrame with community assignments. Defaults to True.
    
    Returns:
        tuple: Depending on return_nodes and return_edges flags, returns:
            - nx.Graph: The input graph with community assignments added as node attributes
            - gpd.GeoDataFrame: Nodes GeoDataFrame with community assignments (if return_nodes=True)
            - gpd.GeoDataFrame: Edges GeoDataFrame with community assignments (if return_edges=True)
    """
    # print(f"Starting community detection using {algorithm} algorithm...")
    start = time.time()

    # If no graph provided, use the one from the object
    if graph is None:
        if not network:
            raise ValueError("No network data found. Please run get_street_network() first.")
        graph = network[0]

    # Make a copy of the graph to avoid modifying the original
    G = graph.copy()

    # Handle weight normalization
    if normalize and weight in nx.get_edge_attributes(G, weight):
        # Normalize weights to [0, 1] range
        weights = np.array(list(nx.get_edge_attributes(G, weight).values()))
        min_weight, max_weight = np.min(weights), np.max(weights)

        if min_weight != max_weight:  # Avoid division by zero
            for u, v, d in G.edges(data=True):
                if weight in d:
                    # For length, shorter edges should have higher weights
                    # For other attributes, you might want to reverse this
                    if weight == 'length':
                        d['weight'] = 1 - ((d[weight] - min_weight) / (max_weight - min_weight))
                    else:
                        d['weight'] = (d[weight] - min_weight) / (max_weight - min_weight)
        else:
            # If all weights are the same, set uniform weights
            for u, v, d in G.edges(data=True):
                d['weight'] = 1.0
    else:
        # If no normalization or no weight attribute, use uniform weights
        for u, v, d in G.edges(data=True):
            d['weight'] = 1.0

    # Community detection
    if algorithm == 'louvain':
        try:
            partition = nx.community.louvain_communities(nx.Graph(G),
                                                         weight='weight',
                                                         resolution=resolution)
            # convert list of sets to dict
            partition = {node: comm for comm, nodes in enumerate(partition) for node in nodes}

        except Exception as e:
            print("Error in Louvain algorithm:", e)

    elif algorithm == 'leiden':
        # Convert NetworkX graph to igraph
        g_ig = ig.Graph(directed=False)
        g_ig.add_vertices(list(G.nodes()))
        # make edges have in range vertex index
        edges = [(list(G.nodes()).index(u), list(G.nodes()).index(v)) for u, v in G.edges()]
        g_ig.add_edges(edges)
        # g_ig.add_edges(list(G.edges()))
        g_ig.es['weight'] = list(nx.get_edge_attributes(G, 'weight').values())

        # Run Leiden algorithm
        partition = leidenalg.find_partition(
            g_ig,
            leidenalg.ModularityVertexPartition,
            weights='weight',
            # resolution_parameter=resolution
        )
        node_mapping = {v.index: v['name'] if 'name' in v.attributes() else v.index for v in g_ig.vs}
        # Convert igraph partition to NetworkX-compatible format
        partition = {node_mapping[node]: i for i, community in enumerate(partition) for node in community}


    # elif algorithm == 'infomap':
    #     try:
    #         # Create an Infomap instance
    #         im = infomap.Infomap("--two-level")
    #         # Add nodes and links to the network
    #         for node in G.nodes():
    #             im.add_node(node, str(node), teleportation_weight=1.0)
    #         for u, v, data in G.edges(data=True):
    #             im.add_link(u, v, data.get('weight', 1.0))

    #         # Run the algorithm
    #         im.run()

    #         # Get the result
    #         partition = {}
    #         for node, module in im.get_modules().items():
    #             partition[node] = module

    #     except ImportError:
    #         print("infomap package not found.")

    elif algorithm == 'label_propagation':
        partition = nx.algorithms.community.label_propagation.label_propagation_communities(nx.Graph(G))
        # Convert to dictionary format
        partition_dict = {}
        for i, community in enumerate(partition):
            for node in community:
                partition_dict[node] = i
        partition = partition_dict

    elif algorithm == 'girvan_newman':
        # Convert to undirected graph for Girvan-Newman
        G_undirected = nx.Graph(G)

        # Remove weak edges based on threshold
        if threshold > 0:
            edges_to_remove = [(u, v) for u, v, d in G_undirected.edges(data=True)
                               if d.get('weight', 1.0) < threshold]
            G_undirected.remove_edges_from(edges_to_remove)

        # Run Girvan-Newman algorithm
        communities_iter = nx.algorithms.community.girvan_newman(G_undirected)

        # Get the first level of communities
        communities = next(communities_iter)

        # Convert to dictionary format
        partition = {}
        for i, community in enumerate(communities):
            for node in community:
                partition[node] = i

    # elif algorithm == 'spectral':
    #     # Convert to undirected graph for spectral clustering
    #     G_undirected = nx.Graph(G)

    #     # Run spectral clustering
    #     partition = nx.algorithms.community.spectral.spectral_clustering(
    #         G_undirected,
    #         k=k,
    #         weight='weight'
    #     )

    #     # Convert to dictionary format
    #     partition = {node: community for node, community in zip(G_undirected.nodes(), partition)}

    elif algorithm == 'asyn_fluidc':
        # Convert to undirected graph for asynchronous fluid communities
        G_undirected = nx.Graph(G)

        # Run asynchronous fluid communities algorithm
        communities = nx.algorithms.community.asyn_fluidc(
            G_undirected,
            k=k,
            max_iter=100,
            seed=42
        )

        # Convert to dictionary format
        partition = {}
        for i, community in enumerate(communities):
            for node in community:
                partition[node] = i

    elif algorithm == 'greedy_modularity':
        # Convert to undirected graph for greedy modularity maximization
        G_undirected = nx.Graph(G)

        # Run greedy modularity maximization algorithm
        communities = nx.algorithms.community.greedy_modularity_communities(
            G_undirected,
            weight='weight'
        )

        # Convert to dictionary format
        partition = {}
        for i, community in enumerate(communities):
            for node in community:
                partition[node] = i

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Please choose from 'louvain', 'leiden', 'label_propagation', 'girvan_newman', 'spectral', 'asyn_fluidc', or 'greedy_modularity'.")

    # Add community assignments to graph nodes
    nx.set_node_attributes(G, partition, 'community')

    # Get the number of communities
    num_communities = len(set(partition.values()))
    # print(f"Detected {num_communities} communities.")

    same_community_nodes = {}
    for k, v in nx.get_node_attributes(G, 'community').items():
        if v not in same_community_nodes:
            same_community_nodes[v] = set()
        same_community_nodes[v].add(k)

    # Get modularity score
    modularity = nx.algorithms.community.modularity(
        nx.Graph(G),
        list(same_community_nodes.values()),
        weight='weight')
    # print(f"Modularity score: {modularity:.4f}")

    # Prepare return values
    results = [G]

    if return_nodes or return_edges:
        # Convert graph to geodataframes
        if dual:
            nodes, edges = osmnx.graph_to_gdfs(G, nodes=True, edges=True, dual=True)
        else:
            nodes, edges = osmnx.graph_to_gdfs(G, nodes=True, edges=True)

        # Add community assignments to nodes
        nodes['community'] = nodes.index.map(lambda x: partition.get(x, -1))

        # Add color column for visualization
        colormap = cm.get_cmap('tab20', num_communities)
        nodes['color'] = nodes['community'].map(lambda x: '#%02x%02x%02x' %
                                               tuple(int(c*255) for c in colormap(x % num_communities)[:3]))

        if return_edges:
            # Add community assignments to edges based on source node
            # u, v are multi level index, u is level 1, v is level 2. multiple v for a u
            edges['source_community'] = edges.index.get_level_values(0).map(lambda x: partition.get(x, -1))
            edges['target_community'] = edges.index.get_level_values(1).map(lambda x: partition.get(x, -1))
            # If source and target are in the same community, assign that community to the edge
            # Otherwise, assign -1 (boundary edge)
            edges['community'] = edges.apply(
                lambda row: row['source_community'] if row['source_community'] == row['target_community'] else -1,
                axis=1
            )

            # Add color column for visualization
            edges['color'] = edges['community'].map(
                lambda x: '#%02x%02x%02x' % tuple(int(c*255) for c in colormap(x % num_communities)[:3]) if x != -1
                else '#808080'  # Gray color for boundary edges
            )

        # Add results to return list
        if return_nodes:
            results.append(nodes)
        if return_edges:
            results.append(edges)

    # print(f"Community detection completed in {time.time() - start:.2f} seconds.")

    return modularity, num_communities, tuple(results)

def visualize_communities(
        nodes=None,
        edges=None,
        figsize=(15, 15),
        node_size=20,
        edge_width=3.0,
        title=None,
        filename=None,
        show=True,
        boundary_color='#808080',
        background_color='white',
        alpha=0.7,
        legend=True,
        legend_position='best',
        legend_columns=1,
        max_communities_legend=10,
        cmap='tab20'):
    """Function to visualize communities detected in the network.
    
    Args:
        nodes (gpd.GeoDataFrame, optional): Nodes GeoDataFrame with community assignments.
        edges (gpd.GeoDataFrame, optional): Edges GeoDataFrame with community assignments.
        figsize (tuple): Figure size as (width, height). Defaults to (15, 15).
        node_size (float): Size of nodes in the plot. Defaults to 15.
        edge_width (float): Width of edges in the plot. Defaults to 1.0.
        title (str, optional): Title of the plot. Defaults to None.
        filename (str, optional): Path to save the figure. If None, figure is not saved. Defaults to None.
        show (bool): Whether to display the plot. Defaults to True.
        boundary_color (str): Color for boundary edges. Defaults to '#808080' (gray).
        background_color (str): Background color of the plot. Defaults to 'white'.
        alpha (float): Transparency of nodes and edges. Defaults to 0.7.
        legend (bool): Whether to show the legend. Defaults to True.
        legend_position (str): Position of the legend. Defaults to 'best'.
        legend_columns (int): Number of columns in the legend. Defaults to 1.
        max_communities_legend (int): Maximum number of communities to show in legend. Defaults to 10.
        cmap (str): Matplotlib colormap for community colors. Defaults to 'tab20'.
    
    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    # Use provided nodes/edges or default to class attributes
    nodes = nodes if nodes is None else nodes
    edges = edges if edges is None else edges

    # Check if community column exists
    community_col = 'community'
    if community_col not in nodes.columns:
        raise ValueError(f"'{community_col}' column not found in nodes. Run detect_communities() first.")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, facecolor=background_color)
    ax.set_facecolor(background_color)

    # Get unique communities and create color mapping
    communities = sorted(nodes[community_col].unique())
    colormap = plt.cm.get_cmap(cmap, len(communities))
    color_dict = {comm: mcolors.rgb2hex(colormap(i)) for i, comm in enumerate(communities)}

    # Create a graph from nodes and edges
    G = nx.Graph()

    # Add nodes with positions
    for idx, row in nodes.iterrows():
        G.add_node(idx, pos=(row.geometry.x, row.geometry.y), community=row[community_col])

    # Add edges
    for _, row in edges.iterrows():
        # Check if source and target nodes exist in G
        if row['source_community'] in G.nodes and row['target_community'] in G.nodes:
            source_comm = G.nodes[row['source_community']]
            target_comm = G.nodes[row['target_community']]

            # Decide if it's a boundary edge
            is_boundary = source_comm != target_comm
            edge_color = boundary_color if is_boundary else color_dict[source_comm]
            edge_alpha = alpha if not is_boundary else alpha * 0.8

            G.add_edge(row['source'], row['target'],
                       color=edge_color,
                       alpha=edge_alpha,
                       weight=edge_width)

    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    # Draw nodes colored by community
    for comm in communities:
        node_list = [node for node, attr in G.nodes(data=True) if attr['community'] == comm]
        nx.draw_networkx_nodes(G, pos,
                              nodelist=node_list,
                              node_color=color_dict[comm],
                              node_size=node_size,
                              alpha=alpha,
                              ax=ax)

    # Draw edges
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]
    edge_alphas = [G[u][v]['alpha'] for u, v in G.edges()]
    edge_widths = [G[u][v]['weight'] for u, v in G.edges()]

    nx.draw_networkx_edges(G, pos,
                          edge_color=edge_colors,
                          width=edge_widths,
                          alpha=edge_alphas,
                          ax=ax)

    # Add legend if requested
    if legend and communities:
        # Limit the number of communities in legend if there are too many
        legend_comms = communities[:max_communities_legend]
        if len(communities) > max_communities_legend:
            legend_comms.append('...')

        legend_handles = [plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=color_dict[comm] if comm != '...' else 'gray',
                                    markersize=10, label=f"Community {comm}")
                         for comm in legend_comms]

        # Add boundary edge to legend
        legend_handles.append(plt.Line2D([0], [0], color=boundary_color,
                                        lw=edge_width*1.5, label='Boundary Edge'))

        fontP = FontProperties()
        fontP.set_size('xx-small')
        plt.legend(handles=legend_handles,
                #   loc=legend_position,
                  ncol=legend_columns,
                  bbox_to_anchor=(1.05, 1),
                  loc='upper left',
                #   prop=fontP
                  frameon=True,
                  framealpha=0.8,
                  title="Communities")

    # Remove axes
    ax.set_axis_off()

    # Set equal aspect ratio
    ax.set_aspect('equal')

    # Set title if provided
    if title:
        plt.title(title, fontsize=14)

    # Tight layout
    # plt.tight_layout()

    # Save figure if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    # Show or close plot
    if show:
        plt.show()
    else:
        plt.close()

    return fig, ax


if __name__ == "__main__":
    import pickle
    saved_data = pickle.load(
        open("/home/srg/projects/urbanity/saved_data/Chiang Mai.pkl", "rb"))
    network, intersections, streets = saved_data["network"], saved_data["intersection"], saved_data["street"]
    # modularity, results = detect_communities(network)
    # print(results)
    # vis
    # visualize_communities(*results[1:], show=True)

    # Find best algorithm
    algorithms = ['louvain', 'leiden', 'label_propagation', 'girvan_newman', 'asyn_fluidc', 'greedy_modularity']
    modularity_scores = {}
    for alg in algorithms:
        modularity, results = detect_communities(network, algorithm=alg)
        modularity_scores[alg] = modularity
    print(modularity_scores)
    best_algorithm = max(modularity_scores, key=modularity_scores.get)
    print(f"Best algorithm: {best_algorithm} (Modularity: {modularity_scores[best_algorithm]:.4f})")
