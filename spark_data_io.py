# %%
import os
os.environ['JAVA_HOME'] ="/usr/lib/jvm/java-21-openjdk-amd64"
os.environ['SPARK_HOME'] = "/home/srg/projects/urbanity/.venv/lib/python3.10/site-packages/pyspark/"
os.environ['LD_LIBRARY_PATH'] += "/home/srg/projects/urbanity/.venv/lib/python3.10/site-packages/"
!echo $SPARK_HOME

# %%
import os
import json
from pyspark.sql import SparkSession
from pyspark.errors import AnalysisException
from sedona.register import SedonaRegistrator
from sedona.utils import SedonaKryoRegistrator, KryoSerializer
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import functions as F
import shapely
from shapely.geometry import shape, Point, LineString, MultiPolygon
import geopandas as gpd
from pyspark.sql import Row

# %%
DATA_FOLDER = "data"  # Folder containing the GeoJSON files
WAREHOUSE_DIR = "spark_data/spark_warehouse"
CHECKPOINT_DIR = "spark_data/spark_checkpoints"
PARQUET_OUTPUT_DIR = "spark_data/parquet_data" 

# %%
spark = (
    SparkSession.builder.appName("urbanity")
    .config("spark.serializer", KryoSerializer.getName)
    .config("spark.kryo.registrator", SedonaKryoRegistrator.getName)
    .config("spark.sql.warehouse.dir", WAREHOUSE_DIR)
    .config("spark.checkpoint.dir", CHECKPOINT_DIR)
    .config("spark.executor.memory", "11g")
    .config("spark.driver.memory", "11g")
    .config("spark.memory.offHeap.enabled", "true")
    .config("spark.memory.offHeap.size", "6g")
    .config("spark.executor.cores", "4")
    .config("spark.driver.cores", "4")
    .config("spark.sql.extensions", "org.apache.sedona.sql.SedonaSqlExtensions")
    .config(

        "spark.jars.packages",
        "org.apache.sedona:sedona-spark-shaded-3.5_2.12:1.7.1,"
        "org.datasyslab:geotools-wrapper:1.7.1-28.5",
    )
    .enableHiveSupport()
    .getOrCreate()
)
# SedonaRegistrator.registerAll(spark)
try:
    spark.sql("CREATE TEMPORARY FUNCTION ST_Point AS 'org.apache.sedona.sql.function.ST_Point'")
    spark.sql("CREATE TEMPORARY FUNCTION ST_Distance AS 'org.apache.sedona.sql.function.ST_Distance'")
    spark.sql("CREATE TEMPORARY FUNCTION ST_GeomFromWKT AS 'org.apache.sedona.sql.function.ST_GeomFromWKT'")
    spark.sql("CREATE TEMPORARY FUNCTION ST_Transform AS 'org.apache.sedona.sql.function.ST_Transform'")
    spark.sql("CREATE TEMPORARY FUNCTION ST_Buffer AS 'org.apache.sedona.sql.function.ST_Buffer'")
    spark.sql("CREATE TEMPORARY FUNCTION ST_Contains AS 'org.apache.sedona.sql.function.ST_Contains'")
    spark.sql("CREATE TEMPORARY FUNCTION ST_Within AS 'org.apache.sedona.sql.function.ST_Within'")
except AnalysisException:
    pass


# %%
node_feats = {"node_id": 0,
              "osmid": 10719909687,
              "x": -70.6501973,
              "y": -33.4348796,
              "Node Density": 6,
              "Street Length": 1652.418,
              "Degree": 3,
              "Clustering": 0.0,
              "Clustering (Weighted)": 0.0,
              "Closeness Centrality": 0.023,
              "Betweenness Centrality": 0.192,
              "Eigenvector Centrality": 0.144,
              "Katz Centrality": 0.007,
              "PageRank": 0.0,
              "Footprint Proportion": 0.649,
              "Footprint Mean": 1014.613,
              "Footprint Stdev": 1481.443,
              "Perimeter Total": 5141.073,
              "Perimeter Mean": 131.822,
              "Perimeter Stdev": 84.007,
              "Complexity Mean": 24.056,
              "Complexity Stdev": 7.797,
              "Building Count": 39,
              "PopSum": 221,
              "Men": 112,
              "Women": 109,
              "Elderly": 32,
              "Youth": 30,
              "Children": 11,
              "Civic": 0.0,
              "Commercial": 13.0,
              "Entertainment": 0.0,
              "Food": 1.0,
              "Healthcare": 0.0,
              "Institutional": 0.0,
              "Recreational": 0.0,
              "Social": 0.0,
              "Green View Mean": 0.14184198113207547,
              "Green View Stdev": 0.12870901457162831,
              "Sky View Mean": 0.14372169811320756,
              "Sky View Stdev": 0.14098222781191871,
              "Building View Mean": 0.27111084905660376,
              "Building View Stdev": 0.19476669555553533,
              "Road View Mean": 0.1296816037735849,
              "Road View Stdev": 0.091192882852780918,
              "Visual Complexity Mean": 1.7235668388197585,
              "Visual Complexity Stdev": 0.26380973260339896}
edge_feats = {"edge_id": 0,
              "u": 1107212892,
              "v": 1107243596,
              "length": 131.997,
              "Footprint Total": 0.0,
              "Footprint Mean": 0.0,
              "Footprint Stdev": 0.0,
              "Complexity Mean": 0.0,
              "Complexity Stdev": 0.0,
              "Perimeter Total": 0.0,
              "Perimeter Mean": 0.0,
              "Perimeter Stdev": 0.0,
              "Building Count": 0,
              "PopSum": 0,
              "Men": 0,
              "Women": 0,
              "Elderly": 0,
              "Youth": 0,
              "Children": 0,
              "Civic": 0.0,
              "Commercial": 0.0,
              "Entertainment": 0.0,
              "Food": 0.0,
              "Healthcare": 0.0,
              "Institutional": 0.0,
              "Recreational": 0.0,
              "Social": 0.0,
              "Street Image Count": 10.0,
              "Green View Mean": 0.0277,
              "Green View Stdev": 0.012401164819842081,
              "Sky View Mean": 0.55231883289124661,
              "Sky View Stdev": 0.10321743184096205,
              "Building View Mean": 0.0093081012378426163,
              "Building View Stdev": 0.01411058682425821,
              "Road View Mean": 0.24920939086294416,
              "Road View Stdev": 0.17807855887867602,
              "Visual Complexity Mean": 1.2904905719487274,
              "Visual Complexity Stdev": 0.18648414428714907}
subzone_feats = {"index": "The Gap neighbourhood plan",
                 "No. of Nodes": 705.0,
                 "No. of Edges": 1660.0,
                 "Area (km2)": 10.39,
                 "Node density (km2)": 67.83,
                 "Edge density (km2)": 159.7,
                 "Total Length (km)": 170.27,
                 "Mean Length (m) ": 102.57,
                 "Length density (km2)": 16.38,
                 "Mean Degree": 4.71,
                 "Mean Neighbourhood Degree": 2.77,
                 "Civic": 0.0,
                 "Commercial": 9.0,
                 "Entertainment": 0.0,
                 "Food": 19.0,
                 "Healthcare": 0.0,
                 "Institutional": 6.0,
                 "Recreational": 57.0,
                 "Social": 0.0,
                 "Building Footprint (Proportion)": 0.95,
                 "Mean Building Footprint (m2)": 417.76,
                 "Building Footprint St.dev (m2)": 732.29,
                 "Total Building Perimeter (m)": 19113.61,
                 "Mean Building Perimeter (m)": 80.99,
                 "Building Perimeter St.dev (m)": 53.67,
                 "Mean Building Complexity": 18.33,
                 "Building Complexity St.dev": 5.78,
                 "PopSum": 17040.349917037362,
                 "Men": 8364.666811,
                 "Women": 8677.415455,
                 "Elderly": 3395.297829,
                 "Youth": 2395.575296,
                 "Children": 1017.842345,
                 "Green View Mean": 0.3459113924050633,
                 "Green View St.dev": 0.13807907215000009,
                 "Sky View Mean": 0.34963853727144861,
                 "Sky View St.dev": 0.1278105998826562,
                 "Building View Mean": 0.011886075949366999,
                 "Building View St.dev": 0.0165618674591686,
                 "Road View Mean": 0.15407876230661041,
                 "Road View St.dev": 0.036601154243918098,
                 "Visual Complexity Mean": 1.5086368199391609,
                 "Visual Complexity St.dev": 0.0}

# %%
def infer_schema_from_geojson(file_path):
    processed_schema = {}
    # Construct schema fields for properties
    fields = []
    with open(file_path, 'r') as f:
        data = json.load(f)
        if 'features' in data and len(data['features']) > 0:
            feature = data['features'][0]
            properties = feature['properties']
            geometry_type = feature['geometry']['type']
            
            for key, value in properties.items():
                if "subzone" in file_path and key in subzone_feats:
                    value = subzone_feats[key]
                elif "edge" in file_path and key in edge_feats:
                    value = edge_feats[key]
                elif "node" in file_path and key in node_feats:
                    value = node_feats[key]
                processed_schema[key] = type(value)
                # if isinstance(value, int):
                #     fields.append(StructField(key, LongType(), True))
                if isinstance(value, float) or isinstance(value, int):
                    fields.append(StructField(key, DoubleType(), True))
                elif isinstance(value, bool):
                    fields.append(StructField(key, BooleanType(), True))
                else:
                    fields.append(StructField(key, StringType(), True))
            
            # Add geometry fields
            fields.append(StructField("geometry_type", StringType(), True))
            fields.append(StructField("geometry_coordinates", StringType(), True))
            fields.append(StructField("geometry_wkt", StringType(), True))
                
    return StructType(fields), geometry_type

# %%
def process_geojson_file(file_path, data_type):
    """
    Process GeoJSON files and convert to Spark DataFrame with proper schema
    """
    # Infer schema
    schema, geometry_type = infer_schema_from_geojson(file_path)
    if schema is None:
        print(f"Error: Could not infer schema from {file_path}")
        return None
    
    # Read GeoJSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Process features
    rows = []
    for feature in data['features']:
        properties = feature['properties']
        geometry = feature['geometry']
        try:
            if geometry["type"] == "GeometryCollection":
                geometry = geometry['geometries'][0]
        except:
            print(feature)
            raise
        # Create a row with properties
        row = {}
        for key, value in properties.items():
            if isinstance(value, int):
                value = float(value)
            row[key] = value
        

        # Add geometry information
        row["geometry_type"] = geometry['type']
        row["geometry_coordinates"] = json.dumps(geometry['coordinates'])
        
        # Convert geometry to WKT for easier spatial operations
        # shape  handles all Point, LineString, and Multi/Polygon 
        # using "type" in dict, "coordinates" for coords
        geom = shape(geometry) 
        row["geometry_wkt"] = geom.wkt
        rows.append(row)

    # rdd = spark.sparkContext.parallelize(rows)
    # df = spark.createDataFrame(rdd, schema)

    # Create Spark DataFrame
    df = spark.createDataFrame(rows, schema)
    
    # Add file type info for tracking source
    df = df.withColumn("data_source_type", lit(data_type))
    df = df.withColumn("source_file", lit(os.path.basename(file_path)))
    df = df.withColumn("city_name", lit(file_path.split(
        "_nodes")[0].split("_subzones")[0].split("_edges")[0]))
    return df

# TODO: Add city name and city CRS, possibly as new table

# %%
def find_and_process_files():
    """
    Find and process all GeoJSON files in the data folder
    """
    node_dfs = []
    edge_dfs = []
    subzone_dfs = []
    
    # Walk through directory and find all GeoJSON files
    root = "data"
    for file in os.listdir(root):
        # if not file.startswith("Zagreb_subzone.geojson"):
            # continue
        if file.endswith('.geojson'):
            file_path = os.path.join(root, file)
            
            # Determine data type based on filename
            if 'nodes' in file.lower():
                print(f"Processing nodes file: {file}")
                df = process_geojson_file(file_path, "nodes")
                if df is not None:
                    node_dfs.append(df)
                    
            elif 'edges' in file.lower():
                print(f"Processing edges file: {file}")
                df = process_geojson_file(file_path, "edges")
                if df is not None:
                    edge_dfs.append(df)
                    
            elif '_subzone' in file.lower():
                print(f"Processing subzone file: {file}")
                df = process_geojson_file(file_path, "subzones")
                if df is not None:
                    subzone_dfs.append(df)

    # Union all dataframes of each type
    nodes_df = node_dfs[0] if node_dfs else None
    if len(node_dfs) > 1:
        for df in node_dfs[1:]:
            nodes_df = nodes_df.unionByName(df, allowMissingColumns=True)
    
    edges_df = edge_dfs[0] if edge_dfs else None
    if len(edge_dfs) > 1:
        for df in edge_dfs[1:]:
            edges_df = edges_df.unionByName(df, allowMissingColumns=True)
    
    subzones_df = subzone_dfs[0] if subzone_dfs else None
    if len(subzone_dfs) > 1:
        for df in subzone_dfs[1:]:
            subzones_df = subzones_df.unionByName(df, allowMissingColumns=True)
    
    return nodes_df, edges_df, subzones_df

# %%
def prepare_spatial_tables(nodes_df, edges_df, subzones_df):
    """
    Prepare spatial tables with Sedona by adding geometry columns
    """
    # After processing and before saving:
    if nodes_df is not None:
        nodes_df = nodes_df.withColumn("geometry", expr("ST_GeomFromWKT(geometry_wkt)"))
        # convert x,y to sedona geometry
        nodes_df = nodes_df.withColumn("location_coordinate", expr("ST_Point(x, y)"))
    if edges_df is not None:
        edges_df = edges_df.withColumn("geometry", expr("ST_GeomFromWKT(geometry_wkt)"))
    if subzones_df is not None:
        subzones_df = subzones_df.withColumn("geometry", expr("ST_GeomFromWKT(geometry_wkt)"))
    
    return nodes_df, edges_df, subzones_df


# %%
def save_as_parquet_and_create_tables(nodes_df, edges_df, subzones_df):
    """
    Save DataFrames as Parquet files and create Spark tables
    """
    # Create database if it doesn't exist
    spark.sql("CREATE DATABASE IF NOT EXISTS urban_network")
    spark.sql("USE urban_network")
    
    # Save nodes data
    if nodes_df is not None:
        # Repartition to control file size
        nodes_df = nodes_df.repartition(5)
        
        # Save as Parquet for efficient storage and retrieval
        nodes_df.write.mode("overwrite").parquet(f"{PARQUET_OUTPUT_DIR}/nodes")
        
        # Create table for SQL access
        spark.sql("""
            CREATE TABLE IF NOT EXISTS urban_network.nodes
            USING PARQUET
            LOCATION '{}'
        """.format(os.path.abspath(f"{PARQUET_OUTPUT_DIR}/nodes")))
        
        print(f"Nodes data saved as Parquet and table created")
        print(f"Node count: {nodes_df.count()}")
    
    # Save edges data
    if edges_df is not None:
        edges_df = edges_df.repartition(5)
        edges_df.write.mode("overwrite").parquet(f"{PARQUET_OUTPUT_DIR}/edges")
        
        spark.sql("""
            CREATE TABLE IF NOT EXISTS urban_network.edges
            USING PARQUET
            LOCATION '{}'
        """.format(os.path.abspath(f"{PARQUET_OUTPUT_DIR}/edges")))
        
        print(f"Edges data saved as Parquet and table created")
        print(f"Edge count: {edges_df.count()}")
    
    # Save subzones data
    if subzones_df is not None:
        subzones_df = subzones_df.repartition(5)
        subzones_df.write.mode("overwrite").parquet(f"{PARQUET_OUTPUT_DIR}/subzones")
        
        spark.sql("""
            CREATE TABLE IF NOT EXISTS urban_network.subzones
            USING PARQUET
            LOCATION '{}'
        """.format(os.path.abspath(f"{PARQUET_OUTPUT_DIR}/subzones")))
        
        print(f"Subzones data saved as Parquet and table created")
        print(f"Subzone count: {subzones_df.count()}")


# %%
# Find and process files
# spark.sql("REFRESH TABLE urban_network.nodes")
# spark.sql("REFRESH TABLE urban_network.edges")
# spark.sql("REFRESH TABLE urban_network.subzones")

nodes_df, edges_df, subzones_df = find_and_process_files()

# Prepare spatial tables
nodes_df, edges_df, subzones_df = prepare_spatial_tables(nodes_df, edges_df, subzones_df)

# Save as Parquet and create tables
save_as_parquet_and_create_tables(nodes_df, edges_df, subzones_df)

# Show database summary
print("\nDatabase Summary:")
spark.sql("SHOW DATABASES").show()
spark.sql("USE urban_network")
spark.sql("SHOW TABLES").show()

# Sample query to show everything is working
print("\nSample of nodes data:")
spark.sql("SELECT * FROM urban_network.nodes LIMIT 5").show(truncate=False)

print("\nSample of edges data:")
spark.sql("SELECT * FROM urban_network.edges LIMIT 5").show(truncate=False)

print("\nSample of subzones data:")
spark.sql("SELECT * FROM urban_network.subzones LIMIT 5").show(truncate=False)


# %%
# spark.sql("DROP TABLE IF EXISTS urban_network.subzones")

# %%
spark.sql("SHOW TABLES IN urban_network").show()
