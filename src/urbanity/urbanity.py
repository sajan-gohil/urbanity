# import base packages
from ipaddress import collapse_addresses
import os
import json
import time
import math
import glob
import warnings
import requests
from unittest import skip
import pkg_resources
from webbrowser import get

# import module functions and classes
from .utils import get_country_centroids, finetune_poi, get_available_precomputed_network_data
from .geom import *
from .building import *
from .population import get_population_data, get_tiled_population_data, raster2gdf
from .topology import compute_centrality, merge_nx_property, merge_nx_attr

# import external functions and classes
import numpy as np
import networkx as nx
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, box
from shapely.errors import ShapelyDeprecationWarning
import ipyleaflet
from ipyleaflet import basemaps, basemap_to_tiles, Icon, Marker, LayersControl, LayerGroup, DrawControl, FullScreenControl, ScaleControl, LocalTileLayer, GeoData
import pyrosm
from pyrosm import get_data
import networkit

# Catch known warnings from shapely and geopandas
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings('ignore', category=FutureWarning)

# Import country coords
country_dict = get_country_centroids()
class Map(ipyleaflet.Map):

    def __init__(self, country: str = None, **kwargs):
        """Instantiates a map object that inherits from ipyleaflet.Map. 

        Args:
            country (str, optional): Name of country to position map view. Defaults to None.
        """        
        self.bbox = None
        self.polygon_bounds = None

        if os.path.isdir('./data'):
            self.directory = "./data"
        else:
            os.makedirs('./data')
            self.directory = "./data"
        

        super().__init__(**kwargs)
    
        if country:
            try:
                self.center = country_dict[country]['coords']
            except KeyError as err:
                print(f"KeyError: {err}. Please manually input center coordinates by passing longitude and latitude information to the `center` argument.")
            finally:
                self.country = country
        
        if 'zoom' not in kwargs:
            self.zoom = 11
        
        # Set default attributes
        if 'layout' not in kwargs:
            self.layout.height = "400px"
            self.layout.width = "600px"

        self.attribution_control = False

        # Add controls
        self.add_control(FullScreenControl())
        self.add_control(LayersControl(position='topright'))

        def handle_draw(target, action: str, geo_json: dict):
            print(action)
            print(geo_json)

        dc = (DrawControl(rectangle={'shapeOptions':{'color':"#a52a2a"}},
                                polyline = {"shapeOptions": {"color": "#6bc2e5", "weight": 2, "opacity": 1.0}},
                                polygon = {"shapeOptions": {"fillColor": "#eba134", "color": "#000000", "fillOpacity": 0.5, "weight":2},
                                            "drawError": {"color": "#dd253b", "message": "Delete and redraw"},
                                            "allowIntersection": False},
                                )
                    )
        dc.on_draw(handle_draw)
        self.add(dc)
        self.add_control(ScaleControl(position='bottomleft'))
        

        self.add_layer(basemap_to_tiles(basemaps.CartoDB.VoyagerNoLabels))
        self.add_layer(LocalTileLayer(path="http://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}", name = 'Google Streets'))
        self.add_layer(LocalTileLayer(path="http://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", name = 'Google Hybrid'))
        self.add_layer(LocalTileLayer(path="http://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}", name = 'Google Terrain'))

    def add_bbox(self, show: bool = False, remove: bool = True):
        """Specifies drawn bounding box as geographic extent.

        Args:
            show (bool, optional): If True, creates another map view. Defaults to False.
            remove (bool, optional): If True, removes drawn bounding box from map after adding it as an attribute. Defaults to True.
        """        

        t_index = None
        
        # Find DrawControl layer index
        for i, control in enumerate(self.controls):
            if isinstance(control, DrawControl):
                t_index = i

        if self.controls[t_index].last_action == '':
            print("No bounding box/polygon found. Please draw on map.")
            if show == True:
                display(self)

        elif self.controls[t_index].last_action == 'created': 

            lon_list = []
            lat_list = []
            for lon,lat in [*self.controls[t_index].data[0]['geometry']['coordinates'][0]]:
                lon_list.append(lon)
                lat_list.append(lat)
        
            polygon_geom = Polygon(zip(lon_list, lat_list))
            gdf = gpd.GeoDataFrame(index=[0], geometry=[polygon_geom]) 

            # Assign bounding box as self object attribute
            self.polygon_bounds = gdf
            self.polygon_bounds.crs = 'epsg:4326'

            # Remove drawing on self and close display
            if remove == True: 
                print('Assigned bbox to map object. Removing drawn boundary.')
                self.controls[t_index].clear()
            else: 
                print('Assigned bbox map object.')
    
    def add_polygon_boundary(
        self, 
        filepath: str,
        layer_name: str = 'Site', 
        polygon_style: dict = {'style': {'color': 'black', 'fillColor': '#3366cc', 'opacity':0.05, 'weight':1.9, 'dashArray':'2', 'fillOpacity':0.6},
                                         'hover_style': {'fillColor': 'red' , 'fillOpacity': 0.2}},                   
        show: bool = False) -> None:
        """Adds geographical boundary from specified filepath. Accepts .geojson and .shapefile objects.

        Args:
            filepath (str): Filepath to vector file.
            layer_name (str, optional): Layer name to display on map object. Defaults to 'Site'.
            polygon_style (dict, optional): Default visualisation parameters to display geographical layer. Defaults to {'style': {'color': 'black', 'fillColor': '#3366cc', 'opacity':0.05, 'weight':1.9, 'dashArray':'2', 'fillOpacity':0.6}, 'hover_style': {'fillColor': 'red' , 'fillOpacity': 0.2}}.
        """        

        if filepath:
            # GeoJSON string file
            gdf = gpd.read_file(filepath)
        
        # Assign polygon boundary attribute to polygon object
        self.polygon_bounds = gdf

        # Add polygon boundary as map layer
        geo_data = GeoData(geo_dataframe = gdf,
                   style=polygon_style['style'],
                   hover_style=polygon_style['hover_style'],
                   name = layer_name)
        self.add_layer(geo_data)
        
    def remove_polygon_boundary(self) -> None:
        """Removes polygon boundary from map object.
        """        
        polygon_exists = False
        for i in self.layers:
            if isinstance(i, ipyleaflet.leaflet.GeoData):
                polygon_exists = True
        if polygon_exists:
            self.remove_layer(self.layers[len(self.layers)-1])
            print('Polygon bounding layer removed.')
        else:
            print('No polygon layer found on map.')

    def check_osm_buildings(self,
                            location: str,
                            column: str = "index") -> gpd.GeoDataFrame:
        """Function to check the attribute completeness for OSM buildings as implemented in: https://ual.sg/publication/2020-3-dgeoinfo-3-d-asean/

        Args:
            location (str): Accepts city name or country name to obtain OpenStreetMap data.
            column (str): Accepts name of column with zone ID or name or defaults to use index.

        Returns:
            gpd.DataFrame: A geopandas dataframe with attribute completeness for OSM buildings
        """

        # First step - Check if bounding box is defined
        try:
            original_bbox = self.polygon_bounds.iloc[[0]].geometry[0]
            # catch when it hasn't even been defined 
        except (AttributeError, NameError):
            raise Exception('Please delimit a bounding box.')
        
        # Obtain filepath to OSM data
        try:
            fp = get_data(location, directory = self.directory)
            print(f'Getting osm building data for {location}')
        except ValueError:
            fp = get_data(self.country, directory = self.directory)
            print(f'ValueError: No pre-downloaded osm data available for {location}, will instead try for {self.country}.')
        
        # Create dictionary keys based on column elements
        attr_stats = {}
        if column == 'index':
            print('No column name specified, using index as column name.')
            for name in self.polygon_bounds.index:
                attr_stats[name] = {}
        else:
            for name in self.polygon_bounds[column]:
                attr_stats[name] = {}

        # Get individual polygon data
        for i, key in enumerate(attr_stats):
            print(f"Checking building data for: {key}")
            # Set bounding box
            original_bbox = self.polygon_bounds.iloc[[i]].geometry[i]
            
            # Get OSM parser
            osm = pyrosm.OSM(fp, bounding_box=original_bbox)

            # Retrieve buildings
            buildings = osm.get_buildings()
            num_buildings = len(buildings)
            attr_stats[key]['No. of Buildings'] = num_buildings

            # Compute attributes
            num_height = len(buildings[~buildings['height'].isna()]) if ('height' in buildings.columns) else 0
            perc_num_height = round(num_height/num_buildings*100,2) if ('height' in buildings.columns) else 0
            attr_stats[key]['No. w/ Height'] = num_height
            attr_stats[key]['Perc w/ Height'] = perc_num_height

            num_levels = len(buildings[~buildings['building:levels'].isna()]) if ('building:levels' in buildings.columns) else 0
            perc_num_levels = round(num_levels/num_buildings*100,2) if ('building:levels' in buildings.columns) else 0
            attr_stats[key]['No. w/ Levels'] = num_levels
            attr_stats[key]['Perc w/ Levels'] = perc_num_levels

        df = pd.DataFrame(attr_stats).transpose()
        gdf = gpd.GeoDataFrame(data=df, crs=self.polygon_bounds.crs, geometry = self.polygon_bounds['geometry'])
        return gdf

    def check_population(self,
                        year: int = 2020) -> [dict, gpd.GeoDataFrame]:
        """Function to check the correspondence of Meta's high resolution population counts (30m) with WorldPop (100m) UN-Adjusted dataset:
        WorldPop (www.worldpop.org - School of Geography and Environmental Science, University of Southampton; Department of Geography and Geosciences, 
        University of Louisville; Departement de Geographie, Universite de Namur) and Center for International Earth Science Information Network (CIESIN), 
        Columbia University (2018). Global High Resolution Population Denominators Project - Funded by The Bill and Melinda Gates Foundation (OPP1134076).

        Args:
            year (int): Specific year to extract World Population data.

        Returns:
            dict: A dictionary consisting of evaluation metrics 
            gpd.DataFrame: A geopandas dataframe with attribute completeness for OSM buildings
        """

        # Get Worldpop .tiff address from ISO code
        ISO_path = pkg_resources.resource_filename('urbanity', 'map_data/GADM_links.json')
        with open(ISO_path) as file:
          ISO = json.load(file)

        path = f'https://data.worldpop.org/GIS/Population/Global_2000_{year}_Constrained/{year}/BSGM/{ISO[self.country]}/{ISO[self.country].lower()}_ppp_{year}_UNadj_constrained.tif'
        maxar_path = f'https://data.worldpop.org/GIS/Population/Global_2000_{year}_Constrained/{year}/maxar_v1/{ISO[self.country]}/{ISO[self.country].lower()}_ppp_{year}_UNadj_constrained.tif'
        # Check if data folder exists, else create one
        if os.path.isdir('./data'):
            self.directory = "./data"
        else:
            os.makedirs('./data')
            self.directory = "./data"

        # Download Worldpop data for specified `year`
        filename = path.split('/')[-1].replace(" ", "_")  # be careful with file names
        filename_maxar = maxar_path.split('/')[-1].replace(" ", "_")  # be careful with file names
        file_path = os.path.join(self.directory, filename)
        file_path_maxar = os.path.join(self.directory, filename_maxar)

        if not os.path.exists(file_path):
            print('Raster file not found in data folder, proceeding to download.')
            r = requests.get(path, stream=True)
            if r.ok:
                print(f"Saved raster file for {self.country} to: \n", os.path.abspath(file_path))
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024 * 8):
                        if chunk:
                            f.write(chunk)
                            f.flush()
                            os.fsync(f.fileno())
            else:
                r = requests.get(maxar_path, stream=True)
                file_path = file_path_maxar
                print(f"Saved raster file for {self.country} to: \n", os.path.abspath(file_path))
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024 * 8):
                        if chunk:
                            f.write(chunk)
                            f.flush()
                            os.fsync(f.fileno())

        else:
            print('Data found! Proceeding to load population data. ')

        
        # Load Worldpop data
        print('Loading Worldpop Population Data...')
        from_raster_100m = raster2gdf(file_path, zoom=True, boundary=self.polygon_bounds)
        from_raster_100m['grid_id'] = range(len(from_raster_100m))

        # Load Meta Population Data
        print('Loading Meta Population Data...')
        tile_countries_path = pkg_resources.resource_filename('urbanity', "map_data/tiled_data.json")
        with open(tile_countries_path, 'r') as f:
            tile_dict = json.load(f)
                    
        tiled_country = [country[:-13] for country in list(tile_dict.keys())]
        groups = ['PopSum', 'Men', 'Women', 'Elderly','Youth','Children']
        
        # Use non-tiled .csv for small countries
        if self.country not in tiled_country:
            print('Using non-tiled population data.')
            pop_gdf, target_col = get_population_data(self.country, 
                                                        bounding_poly=self.polygon_bounds,
                                                        all_only = True)

        # If big country, use csv and custom tiled population data: (e.g. USA: https://figshare.com/articles/dataset/USA_TILE_POPULATION/21502296)
        elif self.country in tiled_country:
            print('Using tiled population data.')
            pop_gdf, target_col = get_tiled_population_data(self.country, 
                                                              bounding_poly = self.polygon_bounds, 
                                                              all_only=True)
        # Preprocess both population files; Add aggregate meta to Worldpop grids

    
        res_intersection = gpd.overlay(pop_gdf, from_raster_100m, how='intersection')
        aggregate_series = res_intersection.groupby(['grid_id'])[target_col].sum()
        combined = from_raster_100m.merge(aggregate_series, on ='grid_id')
        # Get non-empty cells
        non_empty = combined[combined['value']!=-99999.0].copy()
        
        # Get percentage of correct cells
        perc_hits = (len(combined[((combined['value'] == -99999.0) | (combined['value'] == 0)) & (combined[target_col]==0)]) + 
                        len(combined[(combined['value'] > 0) & (combined[target_col]>0)])) / len(combined)
        
        # Get total population counts
        meta_total = non_empty[target_col].sum()
        worldpop_total = non_empty['value'].sum()
        
        # Get correlation between non-empty cells
        world_meta_corr = non_empty['value'].corr(non_empty[target_col])
        
        # Get absolute difference
        non_empty.loc[:,'deviance'] = non_empty['value'] - non_empty[target_col]
        non_empty.loc[:,'abs_deviance'] = abs(non_empty['value'] - non_empty[target_col])
        mean_absolute_error = non_empty['abs_deviance'].mean()
        
        values_dict = {}
        values_dict['Grids hit percentage'] = perc_hits
        values_dict['Meta population total'] = meta_total
        values_dict['Worldpop population total'] = worldpop_total
        values_dict['Worldpop/meta correlation'] = world_meta_corr
        values_dict['Mean absolute error'] = mean_absolute_error
        
        return values_dict, non_empty

    def get_street_network(
            self, 
            location: str,
            filepath: str = '',
            bandwidth: int = 200,
            network_type: str = 'driving',
            graph_attr: bool = True,
            building_attr: bool = True,
            pop_attr: bool = True,
            poi_attr: bool = True,
            svi_attr: bool = False,
            edge_attr: bool = False,
            get_precomputed_100m: bool = False,
            dual: bool = False) -> [nx.MultiDiGraph, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Function to generate either primal planar or dual (edge) networks. If multiple geometries are provided, 
        network is constructed for only the first entry. Please merge geometries before use.
        Bandwidth (m) can be specified to buffer network, obtaining neighbouring nodes within buffered area of network.
        *_attr arguments can be toggled on or off to allow computation of additional geographic information into networks.

        Args:
            location (str): Accepts city name or country name to obtain OpenStreetMap data.
            filepath (str): If location is not available, user can specify path to osm.pbf file.
            bandwidth (int): Distance to extract information beyond network. Defaults to 200.
            network_type (str): Specified OpenStreetMap transportation mode. Defaults to 'driving'.
            graph_attr (bool): Specifies whether graph metric and topological attributes should be included. Defaults to True.
            building_attr (bool): Specifies whether building morphology attributes should be included. Defaults to True.
            pop_attr (bool): Specifies whether population attributes should be included. Defaults to True.
            poi_attr (bool): Specifies whether points of interest attributes should be included. Defaults to True.
            edge_attr (bool): If True, computes edge attributes (available for buildings, pois, population, and svi). Defaults to True.
            get_precomputed_100m (bool): If True, directly downloads network data from the Global Urban Network Repository instead of computing. Defaults to False.
            dual (bool): If true, creates a dual (edge) network graph. Defaults to False.
            
        Raises:
            Exception: No bounding box or polygon found to construct network.

        Returns:
            nx.MultiDiGraph: A networkx/osmnx primal planar or dual (edge) network with specified attribute information.
            gpd.GeoDataFrame: A geopandas dataframe of network nodes with Point geometry.
            gpd.GeoDataFrame: A geopandas dataframe of network edges with Linestring geometry.
        """ 

        if get_precomputed_100m:   
            try:
                network_dataset = pkg_resources.resource_filename('urbanity', "map_data/network_data.json")
                with open(network_dataset, 'r') as f:
                    network_data = json.load(f)
                nodes = gpd.read_file(network_data[f'{location.title()}_nodes_100m.geojson'])
                edges = gpd.read_file(network_data[f'{location.title()}_edges_100m.geojson'])
                return None,nodes,edges
            except:
                get_available_precomputed_network_data()
                return [None, None, None]

        start = time.time()

        if filepath == '':
            try:
                fp = get_data(location, directory = self.directory)
                print('Creating data folder and downloading osm street data...')
            except ValueError:
                fp = get_data(self.country, directory = self.directory)
                print(f"ValueError: No pre-downloaded osm data available for specified city, will instead try for specified country.")
            except ValueError:
                raise ValueError('No osm data found for specified location.')

            print('Data extracted successfully. Proceeding to construct street network.')
        elif filepath != '':
            fp = filepath
            print('Data found! Proceeding to construct street network.')

        # Project and buffer original polygon to examine nodes outside boundary
        try:
            original_bbox = self.polygon_bounds.geometry[0]
            buffered_tp = buffer_polygon(self.polygon_bounds, bandwidth=bandwidth)
            buffered_bbox = buffered_tp.geometry.values[0]
        # catch when it hasn't even been defined 
        except (AttributeError, NameError):
            raise Exception('Please delimit a bounding box.')

        # Obtain nodes and edges within buffered polygon
        osm = pyrosm.OSM(fp, bounding_box=buffered_bbox)

        nodes, edges = osm.get_network(network_type=network_type, nodes=True)

        # Build networkx graph for pre-processing
        G_buff = osm.to_graph(nodes, edges, graph_type="networkx", force_bidirectional=True, retain_all=True)
        
        # Add great circle length to network edges
        G_buff = add_edge_lengths(G_buff)

        # Simplify graph by removing nodes between endpoints and joining linestrings
        G_buff_simple = simplify_graph(G_buff)

        # Identify nodes inside and outside (buffered polygon) of original polygon
        gs_nodes = graph_to_gdf(G_buff_simple, nodes=True)[["geometry"]]
        to_keep = gs_nodes.within(original_bbox)
        to_keep = gs_nodes[to_keep]
        nodes_outside = gs_nodes[~gs_nodes.index.isin(to_keep.index)]
        set_outside = nodes_outside.index

        # Truncate network by edge if all neighbours fall outside original polygon
        nodes_to_remove = set()
        for node in set_outside:
            neighbors = set(G_buff_simple.successors(node)) | set(G_buff_simple.predecessors(node))
            if neighbors.issubset(nodes_outside):
                nodes_to_remove.add(node)
        
        G_buff_trunc = G_buff_simple.copy()
        initial = G_buff_trunc.number_of_nodes()
        G_buff_trunc.remove_nodes_from(nodes_to_remove)

        # Remove unconnected subgraphs
        max_wcc = max(nx.weakly_connected_components(G_buff_trunc), key=len)
        G_buff_trunc = nx.subgraph(G_buff_trunc, max_wcc)

        # Remove self loops
        G_buff_trunc_loop = G_buff_trunc.copy()
        G_buff_trunc_loop.remove_edges_from(nx.selfloop_edges(G_buff_trunc_loop))

        nodes, edges = graph_to_gdf(G_buff_trunc_loop, nodes=True, edges=True)

        # Fill NA and drop incomplete columns
        nodes = nodes.fillna('')
        edges = edges.fillna('')
        nodes = nodes.drop(columns=['osmid','tags','timestamp','version','changeset']).reset_index()
        edges = edges.reset_index()[['u','v','length','geometry']]

        # Assign unique IDs
        nodes['node_id'] = nodes.index
        nodes = nodes[['node_id','osmid', 'x', 'y', 'geometry']]
        edges['edge_id'] = edges.index
        edges = edges[['edge_id', 'u', 'v', 'length','geometry']]

        print(f'Network constructed. Time taken: {round(time.time() - start)}.')

        # If not dual representation graph
        if dual == False:
            
            proj_nodes = project_gdf(nodes)
            proj_edges = project_gdf(edges)

            # Buffer around nodes
            nodes_buffer = proj_nodes.copy()
            nodes_buffer['geometry'] = nodes_buffer.geometry.buffer(bandwidth)
        
            
            if graph_attr:
                # Add Node Density
                res_intersection = proj_nodes.overlay(nodes_buffer, how='intersection')
                res_intersection['Node Density'] = 1
                nodes["Node Density"] = res_intersection.groupby(['osmid_2'])['Node Density'].sum().values

                # Add Street Length
                res_intersection = proj_edges.overlay(nodes_buffer, how='intersection')
                res_intersection['street_len'] = res_intersection.geometry.length
                nodes["Street Length"] = res_intersection.groupby(['osmid'])['street_len'].sum().values
                nodes["Street Length"] = nodes["Street Length"].round(3)

                # Add Degree Centrality, Clustering (Weighted and Unweighted)
                nodes = merge_nx_property(nodes, G_buff_trunc_loop.out_degree, 'Degree')
                nodes = merge_nx_attr(G_buff_trunc_loop, nodes, nx.clustering, 'Clustering')
                nodes = merge_nx_attr(G_buff_trunc_loop, nodes, nx.clustering, 'Clustering (Weighted)', weight='length')


                #  Add Centrality Measures
                nodes = compute_centrality(G_buff_trunc_loop, nodes, networkit.centrality.Closeness, 'Closeness Centrality', False, False)
                nodes = compute_centrality(G_buff_trunc_loop, nodes, networkit.centrality.Betweenness, 'Betweenness Centrality', True)
                nodes = compute_centrality(G_buff_trunc_loop, nodes, networkit.centrality.EigenvectorCentrality, 'Eigenvector Centrality')
                nodes = compute_centrality(G_buff_trunc_loop, nodes, networkit.centrality.KatzCentrality, 'Katz Centrality')
                nodes = compute_centrality(G_buff_trunc_loop, nodes, networkit.centrality.PageRank, 'PageRank', 0.85, 1e-8, networkit.centrality.SinkHandling.NoSinkHandling, True)
            
                print(f'Topologic/metric attributes computed. Time taken: {round(time.time() - start)}.')
            
            # If building_attr is True, compute and add building attributes.
            if building_attr:
                # Get building spatial data and project 
                building = osm.get_buildings()
                building_proj = project_gdf(building)

                # Make geometry type homogeneous (polygons) to to allow overlay operation
                building_polygon = fill_and_expand(building_proj)
                building_polygon = building_polygon.reset_index()

                # Assign unique building id
                building_polygon['bid'] = building_polygon.index
                building_polygon['bid_area'] = building_polygon.geometry.area
                building_polygon['bid_length'] = building_polygon.geometry.length
                building_polygon['bid_complexity'] = building_polygon['bid_length'] / np.sqrt(np.sqrt(building_polygon['bid_area'])) 
                building_polygon = building_polygon[['bid', 'bid_area', 'bid_length', 'bid_complexity', 'geometry']]

                # Compute and add building attributes
                res_intersection = building_polygon.overlay(nodes_buffer, how='intersection')
                # building_set = building_polygon.iloc[list(res_intersection['bid'].unique()),:]
                res_intersection['area'] = res_intersection.geometry.area
                area_series = res_intersection.groupby(['osmid'])['area'].sum()
                total_area = math.pi*bandwidth**2
                area_series = area_series / total_area
                area_series.name = 'Footprint Proportion'
                
                # Obtain proportion 
                nodes = nodes.merge(area_series, on='osmid', how='left')
                
                # Obtain mean area
                mean_series = res_intersection.groupby(['osmid'])['bid_area'].mean()
                mean_series.name = 'Footprint Mean'
                nodes = nodes.merge(mean_series, on='osmid', how='left')

                # Obtain mean area
                std_series = res_intersection.groupby(['osmid'])['bid_area'].std()
                std_series.name = 'Footprint Stdev'
                nodes = nodes.merge(std_series, on='osmid', how='left')

                # Add perimeter
                perimeter_series = res_intersection.groupby(['osmid'])['bid_length'].sum()
                perimeter_series.name = 'Perimeter Total'
                nodes = nodes.merge(perimeter_series, on='osmid', how='left')

                perimeter_mean_series = res_intersection.groupby(['osmid'])['bid_length'].mean()
                perimeter_mean_series.name = 'Perimeter Mean'
                nodes = nodes.merge(perimeter_mean_series, on='osmid', how='left')

                perimeter_std_series = res_intersection.groupby(['osmid'])['bid_length'].std()
                perimeter_std_series.name = 'Perimeter Stdev'
                nodes = nodes.merge(perimeter_std_series, on='osmid', how='left')

                # Add complexity Mean and Std.dev
                compl_mean_series = res_intersection.groupby(['osmid'])['bid_complexity'].mean()
                compl_mean_series.name = 'Complexity Mean'
                nodes = nodes.merge(compl_mean_series, on='osmid', how='left')

                compl_std_series = res_intersection.groupby(['osmid'])['bid_complexity'].std()
                compl_std_series.name = 'Complexity Stdev'
                nodes = nodes.merge(compl_std_series, on='osmid', how='left')

                 # Add counts
                counts_series = res_intersection.groupby(['osmid'])['node_id'].count()
                counts_series.name = 'Building Count'
                nodes = nodes.merge(counts_series, on='osmid', how='left')

                # Add building attributes to node dataframe
                nodes['Footprint Proportion'] = nodes['Footprint Proportion'].replace(np.nan, 0).astype(float).round(3)
                nodes['Footprint Mean'] = nodes['Footprint Mean'].replace(np.nan, 0).astype(float).round(3)
                nodes['Footprint Stdev'] = nodes['Footprint Stdev'].replace(np.nan, 0).astype(float).round(3)
                nodes['Complexity Mean'] = nodes['Complexity Mean'].replace(np.nan, 0).astype(float).round(3)
                nodes['Complexity Stdev'] = nodes['Complexity Stdev'].replace(np.nan, 0).astype(float).round(3)
                nodes['Perimeter Total'] = nodes['Perimeter Total'].replace(np.nan, 0).astype(float).round(3)
                nodes['Perimeter Mean'] = nodes['Perimeter Mean'].replace(np.nan, 0).astype(float).round(3)
                nodes['Perimeter Stdev'] = nodes['Perimeter Stdev'].replace(np.nan, 0).astype(float).round(3)
                nodes['Building Count'] = nodes['Building Count'].replace(np.nan, 0).astype(int)
            
                if edge_attr:
                    building_polygon_centroids = building_polygon.copy()
                    building_polygon_centroids.loc[:,'geometry'] = building_polygon_centroids.geometry.centroid

                    # Assign buildings to nearest edge
                    edge_intersection = gpd.sjoin_nearest(building_polygon_centroids, proj_edges, how='inner', max_distance=50, distance_col = 'Building Distance')

                    # Add footprint sum
                    edge_building_area_sum_series = edge_intersection.groupby(['edge_id'])['bid_area'].sum()
                    edge_building_area_sum_series.name = 'Footprint Total'
                    edges = edges.merge(edge_building_area_sum_series, on='edge_id', how='left')

                    # Add footprint mean
                    edge_building_area_mean_series = edge_intersection.groupby(['edge_id'])['bid_area'].mean()
                    edge_building_area_mean_series.name = 'Footprint Mean'
                    edges = edges.merge(edge_building_area_mean_series, on='edge_id', how='left')

                    # Add footprint std
                    edge_building_area_std_series = edge_intersection.groupby(['edge_id'])['bid_area'].std()
                    edge_building_area_std_series.name = 'Footprint Stdev'
                    edges = edges.merge(edge_building_area_std_series, on='edge_id', how='left')

                    # Add complexity mean
                    edge_building_complexity_mean_series = edge_intersection.groupby(['edge_id'])['bid_complexity'].mean()
                    edge_building_complexity_mean_series.name = 'Complexity Mean'
                    edges = edges.merge(edge_building_complexity_mean_series, on='edge_id', how='left')

                    # Add complexity std
                    edge_building_complexity_std_series = edge_intersection.groupby(['edge_id'])['bid_complexity'].std()
                    edge_building_complexity_std_series.name = 'Complexity Stdev'
                    edges = edges.merge(edge_building_complexity_std_series, on='edge_id', how='left')

                    # Add length sum
                    edge_building_length_sum_series = edge_intersection.groupby(['edge_id'])['bid_length'].sum()
                    edge_building_length_sum_series.name = 'Perimeter Total'
                    edges = edges.merge(edge_building_length_sum_series, on='edge_id', how='left')

                    # Add length mean
                    edge_building_length_mean_series = edge_intersection.groupby(['edge_id'])['bid_length'].mean()
                    edge_building_length_mean_series.name = 'Perimeter Mean'
                    edges = edges.merge(edge_building_length_mean_series, on='edge_id', how='left')

                    # Add length std
                    edge_building_length_std_series = edge_intersection.groupby(['edge_id'])['bid_length'].std()
                    edge_building_length_std_series.name = 'Perimeter Stdev'
                    edges = edges.merge(edge_building_length_std_series, on='edge_id', how='left')

                    # Add buildings counts
                    edge_building_count_series = edge_intersection.groupby(['edge_id'])['Building Distance'].count()
                    edge_building_count_series.name = 'Building Count'
                    edges = edges.merge(edge_building_count_series, on='edge_id', how='left')

                                    # Add building attributes to node dataframe
                    edges['Footprint Total'] = edges['Footprint Total'].replace(np.nan, 0).astype(float).round(3)
                    edges['Footprint Mean'] = edges['Footprint Mean'].replace(np.nan, 0).astype(float).round(3)
                    edges['Footprint Stdev'] = edges['Footprint Stdev'].replace(np.nan, 0).astype(float).round(3)
                    edges['Complexity Mean'] = edges['Complexity Mean'].replace(np.nan, 0).astype(float).round(3)
                    edges['Complexity Stdev'] = edges['Complexity Stdev'].replace(np.nan, 0).astype(float).round(3)
                    edges['Perimeter Total'] = edges['Perimeter Total'].replace(np.nan, 0).astype(float).round(3)
                    edges['Perimeter Mean'] = edges['Perimeter Mean'].replace(np.nan, 0).astype(float).round(3)
                    edges['Perimeter Stdev'] = edges['Perimeter Stdev'].replace(np.nan, 0).astype(float).round(3)
                    edges['Building Count'] = edges['Building Count'].replace(np.nan, 0).astype(int)

                print(f'Building attributes computed. Time taken: {round(time.time() - start)}.')

            # If pop_attr is True, compute and add population attributes.
            if pop_attr:
                tile_countries_path = pkg_resources.resource_filename('urbanity', "map_data/tiled_data.json")
                with open(tile_countries_path, 'r') as f:
                    tile_dict = json.load(f)
                    
                tiled_country = [country[:-13] for country in list(tile_dict.keys())]
                groups = ['PopSum', 'Men', 'Women', 'Elderly','Youth','Children']
                # Use csv for small countries
                if self.country not in tiled_country:
                    print('Using non-tiled population data.')
                    pop_list, target_cols = get_population_data(self.country, 
                                                                bounding_poly=self.polygon_bounds)
                    
                    for i, data in enumerate(zip(pop_list, target_cols)):
                        proj_data = data[0].to_crs(nodes_buffer.crs)
                        res_intersection = proj_data.overlay(nodes_buffer, how='intersection')
                        pop_total_series = res_intersection.groupby(['osmid'])[data[1]].sum()
                        pop_total_series.name = groups[i]
                        nodes = nodes.merge(pop_total_series, on='osmid', how='left')

                        # Add edge attributes
                        if edge_attr:
                            edge_intersection = gpd.sjoin_nearest(proj_data, proj_edges, how='inner', max_distance=50, distance_col = 'Pop Distance')
                            edge_pop_count_series = edge_intersection.groupby(['edge_id'])[data[1]].sum()
                            edge_pop_count_series.name = groups[i]
                            edges = edges.merge(edge_pop_count_series, on='edge_id', how='left')
                            
                    for name in groups:
                        nodes[name] = nodes[name].replace(np.nan, 0).astype(int)
                        if edge_attr:
                                edges[name] = edges[name].replace(np.nan, 0).astype(int)
            
                # If big country, use csv and custom tiled population data: (e.g. USA: https://figshare.com/articles/dataset/USA_TILE_POPULATION/21502296)
                elif self.country in tiled_country:
                    print('Using tiled population data.')
                    pop_list, target_cols = get_tiled_population_data(self.country, bounding_poly = self.polygon_bounds)
                   
                    for i, data in enumerate(zip(pop_list, target_cols)):
                        proj_data = data[0].to_crs(nodes_buffer.crs)
                        res_intersection = proj_data.overlay(nodes_buffer, how='intersection')
                        pop_total_series = res_intersection.groupby(['osmid'])[data[1]].sum()
                        pop_total_series.name = groups[i]
                        nodes = nodes.merge(pop_total_series, on='osmid', how='left')

                        # Add edge attributes
                        if edge_attr:
                            edge_intersection = gpd.sjoin_nearest(proj_data, proj_edges, how='inner', max_distance=50, distance_col = 'Pop Distance')
                            edge_pop_count_series = edge_intersection.groupby(['edge_id'])[data[1]].sum()
                            edge_pop_count_series.name = groups[i]
                            edges = edges.merge(edge_pop_count_series, on='edge_id', how='left')
                            
                    for name in groups:
                        nodes[name] = nodes[name].replace(np.nan, 0).astype(int)
                        if edge_attr:
                                edges[name] = edges[name].replace(np.nan, 0).astype(int)

                print(f'Population attributes computed. Time taken: {round(time.time() - start)}.')   

            # If poi_attr is True, compute and add poi attributes.
            if poi_attr:
                # Load poi information 
                poi_path = pkg_resources.resource_filename('urbanity', "map_data/poi_filter.json")
                with open(poi_path) as poi_filter:
                    poi_filter = json.load(poi_filter)
                
                # Get osm pois based on custom filter
                pois = osm.get_pois(custom_filter = poi_filter['custom_filter'])
                pois = pois.replace(np.nan, '')

                # Relabel amenities to common typology
                def poi_col(amenity, shop, tourism, leisure):
                    value = amenity
                    if amenity == '' and tourism != '':
                        value = 'entertainment'
                    elif amenity == '' and leisure != '':
                        value = 'recreational'
                    elif amenity == '' and shop in poi_filter['food_set']:
                        value = shop
                    elif amenity == '' and shop not in poi_filter['food_set']:
                        value = 'commercial'
                    
                    return value
            
                pois['poi_col'] = pois.apply(lambda row: poi_col(row['amenity'], row['shop'], row['tourism'], row['leisure']), axis=1)
                
                pois = pois[['id', 'osm_type','lon','lat','poi_col','geometry']]

                # Remove amenities that have counts of less than n=5
                pois = finetune_poi(pois, 'poi_col', poi_filter['replace_dict'], n=5)

                pois = project_gdf(pois)
                pois['geometry'] = pois.geometry.centroid

                # Get intersection of amenities with node buffer
                res_intersection = pois.overlay(nodes_buffer, how='intersection')
                poi_series = res_intersection.groupby(['osmid'])['poi_col'].value_counts()
                pois_df = pd.DataFrame(index = poi_series.index, data = poi_series.values).reset_index()
                pois_df = pd.pivot(pois_df, index='osmid', columns='poi_col', values=0).fillna(0)

                col_order = list(nodes.columns)
                cols = list(['civic', 'commercial', 'entertainment', 'food', 'healthcare', 'institutional', 'recreational', 'social'])
                col_order = col_order + cols

                # Add poi attributes to dataframe of nodes
                nodes = nodes.merge(pois_df, on='osmid', how='left')

                for i in cols:
                    if i not in set(nodes.columns):
                        nodes[i] = 0
                    elif i in set(nodes.columns):
                        nodes[i] = nodes[i].replace(np.nan, 0)
                    
                
                nodes = nodes[col_order]
                nodes = nodes.rename(columns = {'commercial':'Commercial', 'entertainment':'Entertainment','food':'Food','healthcare':'Healthcare','civic':'Civic', 'institutional':'Institutional', 'recreational':'Recreational', 'social':'Social'})

                if edge_attr:
                    
                    # Assign pois to nearest edge
                    edge_intersection = gpd.sjoin_nearest(pois, proj_edges, how='inner', max_distance=50, distance_col = 'POI Distance')
                    edge_poi_series = edge_intersection.groupby(['edge_id'])['poi_col'].value_counts()
                    edge_pois_df = pd.DataFrame(index = edge_poi_series.index, data = edge_poi_series.values).reset_index()
                    edge_pois_df = pd.pivot(edge_pois_df, index='edge_id', columns='poi_col', values=0).fillna(0)
                    
                    col_order = list(edges.columns)
                    cols = list(['civic', 'commercial', 'entertainment', 'food', 'healthcare', 'institutional', 'recreational', 'social'])
                    col_order = col_order + cols

                    # Add poi attributes to dataframe of nodes
                    edges = edges.merge(edge_pois_df, on='edge_id', how='left')

                    for i in cols:
                        if i not in set(edges.columns):
                            edges[i] = 0
                        elif i in set(edges.columns):
                            edges[i] = edges[i].replace(np.nan, 0)

                    edges = edges[col_order]
                    edges = edges.rename(columns = {'commercial':'Commercial', 'entertainment':'Entertainment','food':'Food','healthcare':'Healthcare','civic':'Civic', 'institutional':'Institutional', 'recreational':'Recreational', 'social':'Social'})
                    
                print(f'Points of interest computed. Time taken: {round(time.time() - start)}.')

            # If svi_attr is True, compute and add svi attributes.
            if svi_attr:
                tile_gdf = get_tile_geometry(buffered_tp)
                tile_gdf = tile_gdf.set_crs(self.polygon_bounds.crs)
                proj_tile_gdf = project_gdf(tile_gdf)

                svi_path = pkg_resources.resource_filename('urbanity', 'svi_data/svi_data.json')
                with open(svi_path, 'r') as f:
                    svi_dict = json.load(f)
                svi_location = location.replace(' ', '')
                svi_data = gpd.read_file(svi_dict[f'{svi_location}.geojson'])
                svi_data = project_gdf(svi_data)

                # Associate each node with respective tile_id and create mapping dictionary
                tile_id_with_nodes = gpd.sjoin(proj_nodes, proj_tile_gdf)
                node_and_tile = {}
                for k,v in zip(tile_id_with_nodes['tile_id'], tile_id_with_nodes['node_id']):
                    node_and_tile[v] = k

                # Spatial intersection of SVI points and node buffers
                res_intersection = svi_data.overlay(nodes_buffer, how='intersection')

                
                # Compute SVI indices
                indicators = ['Green View', 'Sky View', 'Building View', 'Road View', 'Visual Complexity']
                for indicator in indicators:
                    svi_mean_series = res_intersection.groupby(['osmid'])[indicator].mean()
                    svi_mean_series.name = f'{indicator} Mean'
                    nodes = nodes.merge(svi_mean_series, on='osmid', how='left')
                    svi_tile_mean_aggregate = dict(svi_data.groupby(['tile_id'])[indicator].mean())
                    nodes[f'{indicator} Mean'] = nodes.apply(lambda row: replace_nan_with_tile(svi_tile_mean_aggregate, node_and_tile, row[f'{indicator} Mean'],row.node_id), axis=1)


                    svi_std_series = res_intersection.groupby(['osmid'])[indicator].std()
                    svi_std_series.name = f'{indicator} Stdev'
                    nodes = nodes.merge(svi_std_series, on='osmid', how='left')
                    svi_tile_std_aggregate = dict(svi_data.groupby(['tile_id'])[indicator].std())
                    nodes[f'{indicator} Stdev'] = nodes.apply(lambda row: replace_nan_with_tile(svi_tile_std_aggregate, node_and_tile, row[f'{indicator} Stdev'],row.node_id), axis=1)

                if edge_attr:
                    tile_id_with_edges = gpd.sjoin(proj_edges, proj_tile_gdf)
                    edge_and_tile = {}
                    for k,v in zip(tile_id_with_edges['tile_id'], tile_id_with_edges['edge_id']):
                        edge_and_tile[v] = k
                        
                    # Join SVI points to network edges
                    edge_intersection = gpd.sjoin_nearest(svi_data, proj_edges, how='inner', max_distance=50, distance_col = 'SVI Distance')
                   
                    # Add SVI counts
                    edge_svi_count_series = edge_intersection.groupby(['edge_id'])['SVI Distance'].count()
                    edge_svi_count_series.name = 'Street Image Count'
                    edges = edges.merge(edge_svi_count_series, on='edge_id', how='left')
                    edges['Street Image Count'] = edges['Street Image Count'].replace(np.nan, 0)

                    for indicator in indicators:
                        edge_mean_series = edge_intersection.groupby(['edge_id'])[indicator].mean()
                        edge_mean_series.name = f'{indicator} Mean'
                        edges = edges.merge(edge_mean_series, on='edge_id', how='left')
                        svi_tile_mean_aggregate = dict(svi_data.groupby(['tile_id'])[indicator].mean())
                        edges[f'{indicator} Mean'] = edges.apply(lambda row: replace_nan_with_tile(svi_tile_mean_aggregate, edge_and_tile, row[f'{indicator} Mean'],row.edge_id), axis=1)

                        edge_std_series = edge_intersection.groupby(['edge_id'])[indicator].std()
                        edge_std_series.name = f'{indicator} Stdev'
                        edges = edges.merge(edge_std_series, on='edge_id', how='left')
                        svi_tile_mean_aggregate = dict(svi_data.groupby(['tile_id'])[indicator].mean())
                        edges[f'{indicator} Stdev'] = edges.apply(lambda row: replace_nan_with_tile(svi_tile_mean_aggregate, edge_and_tile, row[f'{indicator} Stdev'],row.edge_id), axis=1)


                print(f'SVI attributes computed. Time taken: {round(time.time() - start)}.')
            # Add computed indices to nodes dataframe

            print("Total elapsed time --- %s seconds ---" % round(time.time() - start))
            
            return G_buff_trunc_loop, nodes, edges

        # If dual is True, construct dual graph with midpoint of original edges as nodes and new edges as adjacency between streets.
        elif dual: 
            # First extract dictionary of osmids and lengths for original nodes associated with each edge
            osmid_view = nx.get_edge_attributes(G_buff_trunc_loop, "osmid")
            osmid_dict = {}
            for u,v in set(osmid_view.items()):
                if u not in osmid_dict:
                    osmid_dict[(u[:2])] = v
                else: 
                    osmid_dict[(u[:2])].append(v)

            length_view = nx.get_edge_attributes(G_buff_trunc_loop, "length")
            length_dict = {}
            for u,v in set(length_view.items()):
                if u not in length_dict:
                    length_dict[(u[:2])] = v
                else: 
                    length_dict[(u[:2])].append(v)

            x_dict = nx.get_node_attributes(G_buff_trunc_loop, "x")
            y_dict = nx.get_node_attributes(G_buff_trunc_loop, "y")

            # Create new placeholder graph and add edges as nodes and adjacency links between edges as new edges
            L = nx.empty_graph(0)
            LG = nx.line_graph(G_buff_trunc_loop)
            L.graph['crs'] = 'EPSG:4326'
            for node in set(G_buff_trunc_loop.edges()):
                L.add_node(node, length = length_dict[node], osmids = osmid_dict[node], x = (x_dict[node[0]]+x_dict[node[1]])/2, y = (y_dict[node[0]]+y_dict[node[1]])/2, geometry=Point((x_dict[node[0]]+x_dict[node[1]])/2, (y_dict[node[0]]+y_dict[node[1]])/2))
            for u,v in set(LG.edges()):
                L.add_edge(u[:2],v[:2])

            # Extract nodes and edges GeoDataFrames from graph
            L_nodes, L_edges = graph_to_gdf(L, nodes=True, edges=True, dual=True)
            L_nodes = L_nodes.fillna('')

            if building_attr:
                building_nodes = project_gdf(L_nodes)
                building_nodes['center_bound'] = building_nodes.geometry.buffer(bandwidth)
                building_nodes = building_nodes.set_geometry('center_bound')
                building_nodes = building_nodes.reset_index()
                building_nodes = building_nodes.rename(columns = {'level_0':'from', 'level_1':'to'})
                building_nodes['unique'] = building_nodes.index

                # Compute and add area
                res_intersection = building_nodes.overlay(building_polygon, how='intersection')
                res_intersection['area'] = res_intersection.geometry.area
                area_series = res_intersection.groupby(['unique'])['area'].sum()
                
                # Obtain proportion 
                total_area = math.pi*bandwidth**2
                area_series = area_series / total_area
                area_series = area_series.astype(float)
                area_series.name = 'Footprint Proportion'

                L_nodes['unique'] = list(range(len(L_nodes)))
                L_nodes = L_nodes.join(area_series, on = 'unique')

                # Add perimeter
                res_intersection['perimeter'] = res_intersection.geometry.length
                perimeter_series = res_intersection.groupby(['unique'])['perimeter'].sum()

                perimeter_series.name = 'Perimeter (m)'
                L_nodes = L_nodes.join(perimeter_series, on = 'unique')


                 # Add counts
                counts_series = res_intersection['unique'].value_counts()
                counts_series.name = 'Counts'
                L_nodes = L_nodes.join(counts_series, on = 'unique')
                
                L_nodes['Footprint Proportion'] = L_nodes['Footprint Proportion'].replace(np.nan, 0).astype(float)
                L_nodes['Perimeter (m)'] = L_nodes['Perimeter (m)'].replace(np.nan, 0).astype(float)
                L_nodes['Counts'] = L_nodes['Counts'].replace(np.nan, 0).astype(int)
    

            print("--- %s seconds ---" % round(time.time() - start,3))

            return L, L_nodes, L_edges
        
    def get_point_context(
            self, 
            location: str,
            points: gpd.GeoDataFrame,
            filepath: str = '',
            bandwidth: int = 200,
            building_attr: bool = True,
            pop_attr: bool = True,
            poi_attr: bool = True,
            svi_attr: bool = False) -> gpd.GeoDataFrame:
        """Function to augment a GeoDataFrame of points with urban contextual information.
        Bandwidth (m) controls Euclidean catchment radius to obtain contextual information.
        *_attr arguments can be toggled on or off to allow computation of additional geographic information into networks.

        Args:
            location (str): Accepts city name or country name to obtain OpenStreetMap data.
            points (gpd.GeoDataFrame): A geopandas dataframe with Point geometry.
            filepath (str): If location is not available, user can specify path to osm.pbf file.
            bandwidth (int): Distance to extract information beyond network. Defaults to 200.
            building_attr (bool): Specifies whether building morphology attributes should be included. Defaults to True.
            pop_attr (bool): Specifies whether population attributes should be included. Defaults to True.
            poi_attr (bool): Specifies whether points of interest attributes should be included. Defaults to True.
            svi_attr (bool): Specifies whether street view imagery attributes should be included. Defaults to False. 
            
        Returns:
            gpd.GeoDataFrame: A geopandas dataframe of Point geometry with augmented geospatial contextual indicators.
        """            

        start = time.time()
        if filepath == '':
            try:
                fp = get_data(location, directory = self.directory)
                print('Creating data folder and downloading osm street data...')
            except ValueError:
                fp = get_data(self.country, directory = self.directory)
                print(f"ValueError: No pre-downloaded osm data available for specified city, will instead try for specified country.")
            except ValueError:
                raise ValueError('No osm data found for specified location.')

            print('Data extracted successfully. Proceeding to extract contextual attributes for point locations.')
        elif filepath != '':
            fp = filepath
            print('Data found! Proceeding to extract contextual attributes for point locations.')

        # Get the projected and buffered bounding box of points
        xmin, ymin, xmax, ymax = points.geometry.total_bounds
        geom = box(xmin, ymin, xmax, ymax)
        original_bbox = gpd.GeoDataFrame(data=None, crs = 'epsg:4326', geometry = [geom])
        buffered_tp = buffer_polygon(original_bbox, bandwidth=bandwidth)
        buffered_bbox = buffered_tp.geometry.values[0]

        # Obtain nodes and edges within buffered polygon
        osm = pyrosm.OSM(fp, bounding_box=buffered_bbox)
        proj_points = project_gdf(points)

        # Buffer around points
        points_buffer = proj_points.copy()
        points_buffer['geometry'] = points_buffer.geometry.buffer(bandwidth)
        
        # If building_attr is True, compute and add building attributes.
        if building_attr:
            # Get building spatial data and project 
            building = osm.get_buildings()
            building_proj = project_gdf(building)

            # Make geometry type homogeneous (polygons) to to allow overlay operation
            building_polygon = fill_and_expand(building_proj)
            building_polygon = building_polygon.reset_index()

            # Assign unique building id
            building_polygon['bid'] = building_polygon.index
            building_polygon['bid_area'] = building_polygon.geometry.area
            building_polygon['bid_length'] = building_polygon.geometry.length
            building_polygon['bid_complexity'] = building_polygon['bid_length'] / np.sqrt(np.sqrt(building_polygon['bid_area'])) 
            building_polygon = building_polygon[['bid', 'bid_area', 'bid_length', 'bid_complexity', 'geometry']]

            # Compute and add building attributes
            res_intersection = building_polygon.overlay(points_buffer, how='intersection')
            # building_set = building_polygon.iloc[list(res_intersection['bid'].unique()),:]
            res_intersection['area'] = res_intersection.geometry.area
            area_series = res_intersection.groupby(['oid'])['area'].sum()
            total_area = math.pi*bandwidth**2
            area_series = area_series / total_area
            area_series.name = 'Footprint Proportion'
            
            # Obtain proportion 
            points = points.merge(area_series, on='oid', how='left')
            
            # Obtain mean area
            mean_series = res_intersection.groupby(['oid'])['bid_area'].mean()
            mean_series.name = 'Footprint Mean'
            points = points.merge(mean_series, on='oid', how='left')

            # Obtain mean area
            std_series = res_intersection.groupby(['oid'])['bid_area'].std()
            std_series.name = 'Footprint Stdev'
            points = points.merge(std_series, on='oid', how='left')

            # Add perimeter
            perimeter_series = res_intersection.groupby(['oid'])['bid_length'].sum()
            perimeter_series.name = 'Perimeter Total'
            points = points.merge(perimeter_series, on='oid', how='left')

            perimeter_mean_series = res_intersection.groupby(['oid'])['bid_length'].mean()
            perimeter_mean_series.name = 'Perimeter Mean'
            points = points.merge(perimeter_mean_series, on='oid', how='left')

            perimeter_std_series = res_intersection.groupby(['oid'])['bid_length'].std()
            perimeter_std_series.name = 'Perimeter Stdev'
            points = points.merge(perimeter_std_series, on='oid', how='left')

            # Add complexity Mean and Std.dev
            compl_mean_series = res_intersection.groupby(['oid'])['bid_complexity'].mean()
            compl_mean_series.name = 'Complexity Mean'
            points = points.merge(compl_mean_series, on='oid', how='left')

            compl_std_series = res_intersection.groupby(['oid'])['bid_complexity'].std()
            compl_std_series.name = 'Complexity Stdev'
            points = points.merge(compl_std_series, on='oid', how='left')

            # Add counts
            counts_series = res_intersection.groupby(['oid'])['oid'].count()
            counts_series.name = 'Building Count'
            points = points.merge(counts_series, on='oid', how='left')

            # Add building attributes to node dataframe
            points['Footprint Proportion'] = points['Footprint Proportion'].replace(np.nan, 0).astype(float).round(3)
            points['Footprint Mean'] = points['Footprint Mean'].replace(np.nan, 0).astype(float).round(3)
            points['Footprint Stdev'] = points['Footprint Stdev'].replace(np.nan, 0).astype(float).round(3)
            points['Complexity Mean'] = points['Complexity Mean'].replace(np.nan, 0).astype(float).round(3)
            points['Complexity Stdev'] = points['Complexity Stdev'].replace(np.nan, 0).astype(float).round(3)
            points['Perimeter Total'] = points['Perimeter Total'].replace(np.nan, 0).astype(float).round(3)
            points['Perimeter Mean'] = points['Perimeter Mean'].replace(np.nan, 0).astype(float).round(3)
            points['Perimeter Stdev'] = points['Perimeter Stdev'].replace(np.nan, 0).astype(float).round(3)
            points['Building Count'] = points['Building Count'].replace(np.nan, 0).astype(int)
        
            print(f'Building morphology attributes computed. Time taken: {round(time.time() - start)}.')   

            # If pop_attr is True, compute and add population attributes.
        if pop_attr:
            tile_countries_path = pkg_resources.resource_filename('urbanity', "map_data/tiled_data.json")
            with open(tile_countries_path, 'r') as f:
                tile_dict = json.load(f)
                
            tiled_country = [country[:-13] for country in list(tile_dict.keys())]
            groups = ['PopSum', 'Men', 'Women', 'Elderly','Youth','Children']
            # Use csv for small countries
            if self.country not in tiled_country:
                print('Using non-tiled population data.')
                pop_list, target_cols = get_population_data(self.country, 
                                                            bounding_poly=buffered_tp)
                
                for i, data in enumerate(zip(pop_list, target_cols)):
                    proj_data = data[0].to_crs(points_buffer.crs)
                    res_intersection = proj_data.overlay(points_buffer, how='intersection')
                    pop_total_series = res_intersection.groupby(['oid'])[data[1]].sum()
                    pop_total_series.name = groups[i]
                    points = points.merge(pop_total_series, on='oid', how='left')

                        
                for name in groups:
                    points[name] = points[name].replace(np.nan, 0).astype(int)
        
            # If big country, use csv and custom tiled population data: (e.g. USA: https://figshare.com/articles/dataset/USA_TILE_POPULATION/21502296)
            elif self.country in tiled_country:
                print('Using tiled population data.')
                pop_list, target_cols = get_tiled_population_data(self.country, bounding_poly = buffered_tp)
                
                for i, data in enumerate(zip(pop_list, target_cols)):
                    proj_data = data[0].to_crs(points_buffer.crs)
                    res_intersection = proj_data.overlay(points_buffer, how='intersection')
                    pop_total_series = res_intersection.groupby(['oid'])[data[1]].sum()
                    pop_total_series.name = groups[i]
                    points = points.merge(pop_total_series, on='oid', how='left')
                        
                for name in groups:
                    points[name] = points[name].replace(np.nan, 0).astype(int)

            print(f'Population attributes computed. Time taken: {round(time.time() - start)}.')   

            # If poi_attr is True, compute and add poi attributes.
        if poi_attr:
            # Load poi information 
            poi_path = pkg_resources.resource_filename('urbanity', "map_data/poi_filter.json")
            with open(poi_path) as poi_filter:
                poi_filter = json.load(poi_filter)
            
            # Get osm pois based on custom filter
            pois = osm.get_pois(custom_filter = poi_filter['custom_filter'])
            pois = pois.replace(np.nan, '')

            # Relabel amenities to common typology
            def poi_col(amenity, shop, tourism, leisure):
                value = amenity
                if amenity == '' and tourism != '':
                    value = 'entertainment'
                elif amenity == '' and leisure != '':
                    value = 'recreational'
                elif amenity == '' and shop in poi_filter['food_set']:
                    value = shop
                elif amenity == '' and shop not in poi_filter['food_set']:
                    value = 'commercial'
                
                return value
        
            pois['poi_col'] = pois.apply(lambda row: poi_col(row['amenity'], row['shop'], row['tourism'], row['leisure']), axis=1)
            
            pois = pois[['id', 'osm_type','lon','lat','poi_col','geometry']]

            # Remove amenities that have counts of less than n=5
            pois = finetune_poi(pois, 'poi_col', poi_filter['replace_dict'], n=5)

            pois = project_gdf(pois)
            pois['geometry'] = pois.geometry.centroid

            # Get intersection of amenities with node buffer
            res_intersection = pois.overlay(points_buffer, how='intersection')
            poi_series = res_intersection.groupby(['oid'])['poi_col'].value_counts()
            pois_df = pd.DataFrame(index = poi_series.index, data = poi_series.values).reset_index()
            pois_df = pd.pivot(pois_df, index='oid', columns='poi_col', values=0).fillna(0)

            col_order = list(points.columns)
            cols = list(['civic', 'commercial', 'entertainment', 'food', 'healthcare', 'institutional', 'recreational', 'social'])
            col_order = col_order + cols

            # Add poi attributes to dataframe of nodes
            points = points.merge(pois_df, on='oid', how='left')

            for i in cols:
                if i not in set(points.columns):
                    points[i] = 0
                elif i in set(points.columns):
                    points[i] = points[i].replace(np.nan, 0)
                
            points = points[col_order]
            points = points.rename(columns = {'commercial':'Commercial', 'entertainment':'Entertainment','food':'Food','healthcare':'Healthcare','civic':'Civic', 'institutional':'Institutional', 'recreational':'Recreational', 'social':'Social'})

            print(f'Points of interest computed. Time taken: {round(time.time() - start)}.')

        # If svi_attr is True, compute and add svi attributes.
        if svi_attr:
            tile_gdf = get_tile_geometry(buffered_tp)
            tile_gdf = tile_gdf.set_crs(buffered_tp.crs)
            proj_tile_gdf = project_gdf(tile_gdf)

            svi_path = pkg_resources.resource_filename('urbanity', 'svi_data/svi_data.json')
            with open(svi_path, 'r') as f:
                svi_dict = json.load(f)
            svi_location = location.replace(' ', '')
            svi_data = gpd.read_file(svi_dict[f'{svi_location}.geojson'])
            svi_data = project_gdf(svi_data)

            # Associate each node with respective tile_id and create mapping dictionary
            tile_id_with_nodes = gpd.sjoin(proj_points, proj_tile_gdf)
            node_and_tile = {}
            for k,v in zip(tile_id_with_nodes['tile_id'], tile_id_with_nodes['oid']):
                node_and_tile[v] = k

            # Spatial intersection of SVI points and node buffers
            res_intersection = svi_data.overlay(points_buffer, how='intersection')

            # Compute SVI indices
            indicators = ['Green View', 'Sky View', 'Building View', 'Road View', 'Visual Complexity']
            for indicator in indicators:
                svi_mean_series = res_intersection.groupby(['oid'])[indicator].mean()
                svi_mean_series.name = f'{indicator} Mean'
                points = points.merge(svi_mean_series, on='oid', how='left')
                svi_tile_mean_aggregate = dict(svi_data.groupby(['tile_id'])[indicator].mean())
                points[f'{indicator} Mean'] = points.apply(lambda row: replace_nan_with_tile(svi_tile_mean_aggregate, node_and_tile, row[f'{indicator} Mean'],row.oid), axis=1)


                svi_std_series = res_intersection.groupby(['oid'])[indicator].std()
                svi_std_series.name = f'{indicator} Stdev'
                points = points.merge(svi_std_series, on='oid', how='left')
                svi_tile_std_aggregate = dict(svi_data.groupby(['tile_id'])[indicator].std())
                points[f'{indicator} Stdev'] = points.apply(lambda row: replace_nan_with_tile(svi_tile_std_aggregate, node_and_tile, row[f'{indicator} Stdev'],row.oid), axis=1)

            print(f'SVI attributes computed. Time taken: {round(time.time() - start)}.')
        # Add computed indices to nodes dataframe

        print("Total elapsed time --- %s seconds ---" % round(time.time() - start))
        
        return points

    def get_aggregate_stats(
        self,
        location: str,
        filepath: str = '',
        column: str = None, 
        bandwidth: int = 0,
        get_svi: bool = False,
        network_type: str = 'driving') -> pd.DataFrame:
        """Obtains descriptive statistics for bounding polygon without constructing network. Users can specify bounding polygon either by drawing on the map object, or uploading a geojson/shapefile.
        If geojson/shape file contains multiple geometric objects, descriptive statistics will be returned for all entities. Results are returned in dictionary format. 

        Args:
            location (str): Accepts city name or country name to obtain OpenStreetMap data.
            filepath (str): If location is not available, user can specify path to osm.pbf file.
            column (str): Id or name column to identify zones. If None, uses shapefile index column.
            data_path(str): Accepts path to shapefile or geojson object
            bandwidth (int): Distance (m) to buffer site boundary. Defaults to 0.
            get_svi(bool): If True, includes aggregated SVI indicators.
            network_type (str): Specified OpenStreetMap transportation mode. Defaults to 'driving'.
        Returns:
            pd.DataFrame: Pandas dataframe consisting of aggregate values for each subzone.
        """
        start = time.time()
        if filepath == '':
            try:
                fp = get_data(location, directory = self.directory)
                print('Creating data folder and downloading osm street data...')
            except ValueError:
                fp = get_data(self.country, directory = self.directory)
                print(f"KeyError: No pre-downloaded osm data available for specified city, will instead try for specified country.")
            except ValueError:
                raise ValueError('No osm data found for specified location.')

            print('Data extracted successfully. Proceeding to construct street network.')
        elif filepath != '':
            fp = filepath
            print('Data found! Proceeding to construct street network.')

        print('Data extracted successfully. Computing aggregates from shapefile.')

        # Create dictionary keys based on column elements
        attr_stats = {}
        if column == 'name':
            column = 'name_id'
            self.polygon_bounds.rename(columns={'name':column}, inplace=True)
        try:
            for name in self.polygon_bounds[column]:
                attr_stats[name] = {}
        except KeyError:
            for name in self.polygon_bounds.index:
                attr_stats[name] = {}

        # Load Global Data
        local_crs = project_gdf(self.polygon_bounds).crs

        # Points of Interest
        poi_path = pkg_resources.resource_filename('urbanity', "map_data/poi_filter.json")
        with open(poi_path) as poi_filter:
            poi_filter = json.load(poi_filter)

        def poi_col(amenity, shop, tourism, leisure):
            value = amenity
            if amenity == '' and tourism != '':
                value = 'entertainment'
            elif amenity == '' and leisure != '':
                value = 'recreational'
            elif amenity == '' and shop in poi_filter['food_set']:
                value = shop
            elif amenity == '' and shop not in poi_filter['food_set']:
                value = 'commercial'
            
            return value
        
        # Population
        tile_countries_path = pkg_resources.resource_filename('urbanity', "map_data/tiled_data.json")
        with open(tile_countries_path, 'r') as f:
            tile_dict = json.load(f)
            
        tiled_country = [country[:-13] for country in list(tile_dict.keys())]
        groups = ['PopSum', 'Men', 'Women', 'Elderly','Youth','Children']

        if self.country not in tiled_country:
            pop_list, target_cols = get_population_data(self.country, bounding_poly=self.polygon_bounds)
            for i in range(len(pop_list)):
                pop_list[i] = pop_list[i].to_crs(local_crs)

        if self.country in tiled_country:    
            pop_list, target_cols = get_tiled_population_data(self.country, bounding_poly=self.polygon_bounds)
            for i in range(len(pop_list)):
                pop_list[i] = pop_list[i].to_crs(local_crs)
        
        if get_svi:
            svi_path = pkg_resources.resource_filename('urbanity', 'svi_data/svi_data.json')
            with open(svi_path, 'r') as f:
                svi_dict = json.load(f)
            svi_location = location.replace(' ', '')
            svi_data = gpd.read_file(svi_dict[f'{svi_location}.geojson'])
            svi_data = project_gdf(svi_data)

        # Get individual polygon data
        for i, key in enumerate(attr_stats):

            # Project and buffer original polygon
            proj_gdf = project_gdf(self.polygon_bounds.iloc[[i],:])
            proj_gdf_buffered = proj_gdf.buffer(bandwidth)

            # Logical gate to check if column id is specified.
            if column is None:
                proj_gdf_buffered = gpd.GeoDataFrame(data=['Site'], crs = proj_gdf.crs, geometry = proj_gdf_buffered)
            else: 
                proj_gdf_buffered = gpd.GeoDataFrame(data=proj_gdf[column], crs = proj_gdf.crs, geometry = proj_gdf_buffered)
            
            area = proj_gdf_buffered.geometry.area.values.item() / 1000000
            
            # Obtain pyrosm query object within spatial bounds
            original_bbox = self.polygon_bounds.iloc[[i],:].geometry.values[0]
            buffered_tp = buffer_polygon(self.polygon_bounds.iloc[[i],:], bandwidth=bandwidth)
            buffered_bbox = buffered_tp.geometry.values[0]
            osm = pyrosm.OSM(fp, bounding_box=buffered_bbox)
            nodes, edges = osm.get_network(network_type=network_type, nodes=True)

            # Build networkx graph for pre-processing
            G_buff = osm.to_graph(nodes, edges, graph_type="networkx", force_bidirectional=True, retain_all=True)
            

            # Add great circle length to network edges
            G_buff = add_edge_lengths(G_buff)

            # Simplify graph by removing nodes between endpoints and joining linestrings
            G_buff_simple = simplify_graph(G_buff)

            # Identify nodes inside and outside (buffered polygon) of original polygon
            gs_nodes = graph_to_gdf(G_buff_simple, nodes=True)[["geometry"]]
            to_keep = gs_nodes.within(original_bbox)
            to_keep = gs_nodes[to_keep]
            nodes_outside = gs_nodes[~gs_nodes.index.isin(to_keep.index)]
            set_outside = nodes_outside.index

            # Truncate network by edge if all neighbours fall outside original polygon
            nodes_to_remove = set()
            for node in set_outside:
                neighbors = set(G_buff_simple.successors(node)) | set(G_buff_simple.predecessors(node))
                if neighbors.issubset(nodes_outside):
                    nodes_to_remove.add(node)
            
            G_buff_trunc = G_buff_simple.copy()
            initial = G_buff_trunc.number_of_nodes()
            G_buff_trunc.remove_nodes_from(nodes_to_remove)

            # Remove unconnected subgraphs
            max_wcc = max(nx.weakly_connected_components(G_buff_trunc), key=len)
            G_buff_trunc = nx.subgraph(G_buff_trunc, max_wcc)

            # Remove self loops
            G_buff_trunc_loop = G_buff_trunc.copy()
            G_buff_trunc_loop.remove_edges_from(nx.selfloop_edges(G_buff_trunc_loop))

            nodes, edges = graph_to_gdf(G_buff_trunc_loop, nodes=True, edges=True)
            nodes = nodes.fillna('')
            
            print(f"Computing aggregate attributes for: {key}")
            
            # Add geometric/metric attributes
            attr_stats[key]["No. of Nodes"] = len(nodes)
            attr_stats[key]["No. of Edges"] = len(edges)
            attr_stats[key]["Area (km2)"] = round(area,2)
            attr_stats[key]["Node density (km2)"] = round(attr_stats[key]["No. of Nodes"] / area, 2)
            attr_stats[key]["Edge density (km2)"]  = round(attr_stats[key]["No. of Edges"] / area, 2)
            attr_stats[key]["Total Length (km)"] = round(G_buff_trunc_loop.size(weight='length')/1000,2)
            attr_stats[key]["Mean Length (m) "] = round(attr_stats[key]["Total Length (km)"] / attr_stats[key]["No. of Edges"] * 1000, 2)
            attr_stats[key]["Length density (km2)"] = round(attr_stats[key]["Total Length (km)"] /area, 2)
            attr_stats[key]["Mean Degree"] = round(2 * attr_stats[key]["No. of Edges"] / attr_stats[key]["No. of Nodes"], 2)
            attr_stats[key]["Mean Neighbourhood Degree"] = round(sum(nx.average_neighbor_degree(G_buff_trunc_loop).values()) / len(nodes), 2)

            # Add Points of Interest
            pois = osm.get_pois(custom_filter = poi_filter['custom_filter'])

            if pois is not None:
                pois = pois.replace(np.nan, '')
                for c in ['shop', 'tourism', 'leisure']:
                    if c not in pois.columns:
                        pois[c] = ''
                poi_set = ['amenity', 'shop', 'tourism', 'leisure']
                for i in poi_set:
                    if i in pois.columns:
                        pass
                    else:
                        pois[i] = ''

                pois['poi_col'] = pois.apply(lambda row: poi_col(row['amenity'], row['shop'], row['tourism'], row['leisure']), axis=1)
                pois['geometry'] = pois.geometry.centroid
                pois['lon'] = pois.geometry.x
                pois['lat'] = pois.geometry.y
                
                pois = pois[['id', 'osm_type','lon','lat','poi_col','geometry']]
                pois = finetune_poi(pois, 'poi_col', poi_filter['replace_dict'], n=2)
                
            
                if len(pois) == 0:
                    cols = ['Civic', 'Commercial', 'Entertainment', 'Food', 'Healthcare', 'Institutional', 'Recreational', 'Social']
                    for i in cols:
                        attr_stats[key][i] = 0
                
                else: 
                    pois = project_gdf(pois)
                    pois['geometry'] = pois.geometry.centroid
                    res_intersection = pois.overlay(proj_gdf_buffered, how='intersection')
                    
                    if column is None:
                        poi_series = res_intersection.groupby([0])['poi_col'].value_counts()
                    else: 
                        poi_series = res_intersection.groupby([column])['poi_col'].value_counts()

                    cols = ['Civic', 'Commercial', 'Entertainment', 'Food', 'Healthcare', 'Institutional', 'Recreational', 'Social']
                    for i in cols:
                        attr_stats[key][i] = 0
                    
                    for poi, counts in poi_series.items():
                        attr_stats[key][str.title(poi[1])] = counts

            else:
                cols = ['Civic', 'Commercial', 'Entertainment', 'Food', 'Healthcare', 'Institutional', 'Recreational', 'Social']
                for i in cols:
                    attr_stats[key][i] = 0
                

            # Add Buildings
            building = osm.get_buildings()
            building_proj = project_gdf(building)
            building_polygon = fill_and_expand(building_proj)
            building_polygon = building_polygon.reset_index()
            building_polygon['bid'] = building_polygon.index

            res_intersection = proj_gdf_buffered.overlay(building_polygon, how='intersection')
            building_set = building_polygon.iloc[list(res_intersection['bid'].unique()),:]
            building_area = res_intersection.geometry.area.sum() / 1000000
            attr_stats[key]["Building Footprint (Proportion)"] = round(building_area/attr_stats[key]["Area (km2)"]*100,2)
            attr_stats[key]["Mean Building Footprint (m2)"] = round(building_set.geometry.area.mean(),2)
            attr_stats[key]["Building Footprint St.dev (m2)"] = round(building_set.geometry.area.std(),2)
            attr_stats[key]["Total Building Perimeter (m)"] = round(building_set.geometry.length.sum(), 2)
            attr_stats[key]["Mean Building Perimeter (m)"] = round(building_set.geometry.length.mean(), 2)
            attr_stats[key]["Building Perimeter St.dev (m)"] = round(building_set.geometry.length.std(),2)
            attr_stats[key]["Mean Building Complexity"] = round(np.mean(building_set.geometry.length / np.sqrt(np.sqrt(building_set.geometry.area))),2)
            attr_stats[key]["Building Complexity St.dev"] = round(np.std(building_set.geometry.length / np.sqrt(np.sqrt(building_set.geometry.area))),2)


            # Add Population
            for i in range(len(pop_list)):
                res_intersection = pop_list[i].overlay(proj_gdf_buffered, how='intersection')
                attr_stats[key][groups[i]] = np.sum(res_intersection.iloc[:,0])

            if get_svi:
                # Add SVI
                res_intersection = svi_data.overlay(proj_gdf_buffered, how = 'intersection')

                # Compute SVI indices
                indicators = ['Green View', 'Sky View', 'Building View', 'Road View', 'Visual Complexity']
                for indicator in indicators:
                    attr_stats[key][f'{indicator} Mean'] = res_intersection[indicator].mean()
                    attr_stats[key][f'{indicator} St.dev'] = res_intersection[indicator].std()

        return pd.DataFrame(attr_stats).transpose()

    
    def get_building_network(
            self, 
            location: str,
            network_type: str = 'driving',
            bandwidth: int = 200) -> nx.MultiDiGraph:
        """Generate a four-layer heterogeneous graph network. Nodes correspond to street intersections, mid-point of streets, urban plots, and buildings. 
        Edges between nodes are determined by their spatial relationships (e.g. buildings within urban plots, nearest buildings sharing a common street edge, intersection connecting to street, and streets adjacent to urban plot). 

        Args:
            network_type (str, optional): Specified OpenStreetMap transportation mode. Defaults to 'driving'.

        Raises:
            Exception: No bounding box found. 

        Returns:
            nx.MultiDiGraph: Returns a building network object. 
        """        
        start = time.time()
        print('Creating data folder and downloading osm building data...')
        fp = get_data(self.country, directory = self.directory)
        print('Data extracted successfully. Proceeding to construct building network.')

        # Project and buffer original polygon to examine nodes outside boundary
        try:
            original_bbox = self.polygon_bounds.geometry.values[0]
            buffered_tp = buffer_polygon(self.polygon_bounds, bandwidth=bandwidth)
            buffered_bbox = buffered_tp.geometry.values[0]
        # catch when it hasn't even been defined 
        except (AttributeError, NameError):
            raise Exception('Please delimit a bounding box.')

        # Obtain nodes and edges within buffered polygon
        osm = pyrosm.OSM(fp, bounding_box=buffered_bbox)

        nodes, edges = osm.get_network(network_type=network_type, nodes=True)

        # Build networkx graph for pre-processing
        G_buff = osm.to_graph(nodes, edges, graph_type="networkx", force_bidirectional=True, retain_all=True)
        
        # Add great circle length to network edges
        G_buff = add_edge_lengths(G_buff)

        # Simplify graph by removing nodes between endpoints and joining linestrings
        G_buff_simple = simplify_graph(G_buff)

        # Identify nodes inside and outside (buffered polygon) of original polygon
        gs_nodes = graph_to_gdf(G_buff_simple, nodes=True)[["geometry"]]
        to_keep = gs_nodes.within(original_bbox)
        to_keep = gs_nodes[to_keep]
        nodes_outside = gs_nodes[~gs_nodes.index.isin(to_keep.index)]
        set_outside = nodes_outside.index

        # Truncate network by edge if all neighbours fall outside original polygon
        nodes_to_remove = set()
        for node in set_outside:
            neighbors = set(G_buff_simple.successors(node)) | set(G_buff_simple.predecessors(node))
            if neighbors.issubset(nodes_outside):
                nodes_to_remove.add(node)
        
        G_buff_trunc = G_buff_simple.copy()
        initial = G_buff_trunc.number_of_nodes()
        G_buff_trunc.remove_nodes_from(nodes_to_remove)

        # Remove unconnected subgraphs
        max_wcc = max(nx.weakly_connected_components(G_buff_trunc), key=len)
        G_buff_trunc = nx.subgraph(G_buff_trunc, max_wcc)

        # Remove self loops
        G_buff_trunc_loop = G_buff_trunc.copy()
        G_buff_trunc_loop.remove_edges_from(nx.selfloop_edges(G_buff_trunc_loop))

        nodes, edges = graph_to_gdf(G_buff_trunc_loop, nodes=True, edges=True)

        # Fill NA and drop incomplete columns
        nodes = nodes.fillna('')
        edges = edges.fillna('')
        nodes = nodes.drop(columns=['osmid','tags','timestamp','version','changeset']).reset_index()
        edges = edges.reset_index()[['u','v','length','geometry']]

        # Assign unique IDs
        nodes['node_id'] = nodes.index
        nodes = nodes[['node_id','osmid', 'x', 'y', 'geometry']]
        edges['edge_id'] = edges.index
        edges = edges[['edge_id', 'u', 'v', 'length','geometry']]

        print(f'Network constructed. Time taken: {round(time.time() - start)}.')

        # Get buildings
        overture_buildings = get_overture_buildings(location)
        osm_buildings = osm.get_buildings()

        # Process geometry and attributes for Overture buildings
        overture_geom = preprocess_overture_building_geometry(overture_buildings, minimum_area=30)
        overture_attr = preprocess_overture_building_attributes(overture_geom, return_class_height=False)

        # Process geometry and attributes for Overture buildings
        osm_geom = preprocess_osm_building_geometry(osm_buildings, minimum_area=30)
        osm_attr = preprocess_osm_building_attributes(osm_geom, return_class_height=False)

        # Obtain unique ids for buildings
        overture_attr_uids = assign_numerical_id_suffix(overture_attr, 'overture')
        osm_attr_uids = assign_numerical_id_suffix(osm_attr, 'osm')

        # Merged building and augment with additional attributes from OSM
        merged_building = merge_osm_to_overture_footprints(overture_attr_uids, osm_attr_uids)
        merged_building_attr = extract_attributed_osm_buildings(merged_building, osm_attr_uids, column = 'osm_combined_heights', threshold = 50)

        # Obtain building network
        merged_building_attr_nodes = merged_building_attr.copy()
        merged_building_attr_nodes['centroid'] = merged_building_attr_nodes.geometry.centroid
        merged_building_attr_nodes = merged_building_attr_nodes.set_geometry("centroid")
        merged_building_attr_nodes = merged_building_attr_nodes.to_crs(4326)
        merged_building_attr_nodes['x'] = merged_building_attr_nodes.geometry.x
        merged_building_attr_nodes['y'] = merged_building_attr_nodes.geometry.y
        merged_building_attr_nodes = merged_building_attr_nodes.set_index('building_id')

        print(f'Buildings constructed. Time taken: {round(time.time() - start)}.')

        return nodes, edges, merged_building_attr, merged_building_attr_nodes
        # Create building network graph
        # Get dict of building nodes and their attributes
        id_to_attributes = {}
        for node in set(merged_building_attr_nodes.index):
            id_to_attributes[node] = merged_building_attr_nodes.loc[node].to_dict()

        # Create empty graph and set graph attribute
        B = nx.empty_graph(0)
        B.graph['crs'] = 'EPSG:4326'

        # Add nodes to graph
        for node in set(merged_building_attr_nodes.index):
            B.add_node(node)
        nx.set_node_attributes(B, id_to_attributes)

        # Get adjacency based on spatial intersection
        building_neighbours = {}
        for i,b in zip(merged_building_attr.building_id, merged_building_attr.geometry):
            s = merged_building_attr.intersects(b.buffer(100))
            building_neighbours[i] = merged_building_attr.building_id[s[s]['building_id']].values


        # Add edges
        for node, neighbours in building_neighbours.items():
            for neighbour in neighbours:
                B.add_edge(node, neighbour)

        # Compute euclidean distance between adjacent building centroids
        id_to_x = dict(zip(merged_building_attr_nodes.index, merged_building_attr_nodes.x))
        id_to_y = dict(zip(merged_building_attr_nodes.index, merged_building_attr_nodes.y))


        distance_between_buildings = {}
        for pair in set(B.edges()):
            distance_between_buildings[pair] = great_circle_vec(id_to_x[pair[0]], id_to_y[pair[0]], id_to_x[pair[1]], id_to_y[pair[1]])
        
        # nx.set_edge_attributes(B, distance_between_buildings, 'length')

        # # Identify largest weakly connected component
        # max_wcc = max(nx.connected_components(B), key=len)
        # B_max = nx.subgraph(B, max_wcc)

        # B_nodes, B_edges = graph_to_gdf(B_max, nodes=True, edges=True, dual=True)
        # B_nodes = B_nodes.fillna('')

        # print("--- %s seconds ---" % round(time.time() - start,3))

        # return B_max, B_nodes, B_edges