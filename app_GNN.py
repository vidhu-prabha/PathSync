from flask import Flask, request, jsonify, render_template, redirect, url_for
import pandas as pd
import numpy as np
import snowflake.connector
import googlemaps
import folium
from folium.plugins import MarkerCluster, AntPath
from sqlalchemy import create_engine
import requests
import time
import random
import concurrent.futures
import nest_asyncio
import warnings
from requests.exceptions import ReadTimeout
from sklearn.cluster import KMeans
import os
import torch
import torch.nn as nn 
import torch.optim as optim 
from scipy.spatial.distance import cdist
from IPython.display import IFrame


distance_cache = {}

# Apply asyncio patch
nest_asyncio.apply()

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")
# Initialize Google Maps client (Replace with your actual API key)
google_api_key = 'google_api_key'

weather_api_key = 'weather_api_key'
gmaps = googlemaps.Client(key=google_api_key)
base_url = 'https://maps.googleapis.com/maps/api/directions/json'

app = Flask(__name__)

app.config['DEBUG'] = True

def get_snowflake_connection():
    return snowflake.connector.connect(
        user='user',
        password='password',
        account='account',
        warehouse='WH',
        database='DB',
        schema='SC'
    )

selected_city = 'Lake Hiawatha'

def check_snf_conn():
    conn = get_snowflake_connection()
    cur = conn.cursor()
    cur.execute("SELECT CURRENT_VERSION()")
    data = cur.fetchone()
    print("Snowflake Version:", data[0])
    cur.close()


#*************************     get customer coordinates and facility details    ******************************
def get_addresses_for_city(selected_city):
    print(selected_city)
    conn = get_snowflake_connection()
    addresses = []

    try:
        cursor = conn.cursor()
        query = """
        SELECT CUST_ADDRESS, CUST_CITY, CUST_STATE, CUST_ZIP, CUST_LATITUDE, CUST_LONGITUDE
        FROM CUST_DATA_TB
        WHERE FAC_CITY = %s
        """

        cursor.execute(query, (selected_city,))
        rows = cursor.fetchall()

        for row in rows:
            address = {
                'CUST_ADDRESS': row[0],
                'CUST_CITY': row[1],
                'CUST_STATE': row[2],
                'CUST_ZIP': row[3],
                'CUST_LATITUDE': row[4],
                'CUST_LONGITUDE': row[5]
            }
            addresses.append(address)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cursor.close()
        conn.close()
    
    return addresses
    

def get_facility_details(selected_city):
    print(selected_city)
    conn = get_snowflake_connection()
    
    fac_details = None  # Initialize as None to handle the case where no results are found

    try:
        cursor = conn.cursor()
        query = """
        SELECT FAC_NAME, FAC_ADDRESS, FAC_CITY, FAC_STATE, FAC_ZIP, FAC_LATITUDE, FAC_LONGITUDE, FAC_DISTANCE
        FROM CUST_DATA_TB
        WHERE FAC_CITY = %s
        ORDER BY FAC_DISTANCE  
        LIMIT 1
        """

        cursor.execute(query, (selected_city,))
        row = cursor.fetchone()  

        if row:
            fac_details = {
                'FAC_NAME': row[0],
                'FAC_ADDRESS': row[1],
                'FAC_CITY': row[2],
                'FAC_STATE': row[3],
                'FAC_ZIP': row[4],
                'FAC_LATITUDE': row[5],
                'FAC_LONGITUDE': row[6],
                'FAC_DISTANCE': row[7]
            }
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cursor.close()
        conn.close()
    
    return fac_details


#********************** GNN to solve TSP ***********************
def get_route_from_google_maps(api_key, origin, destinations):
    gmaps = googlemaps.Client(key=api_key)
    
    all_route_points = []
    total_distance = 0
    total_duration = 0
    
    for destination in destinations:
        # Fetch directions from the Google Maps API
        directions_result = gmaps.directions(origin, destination)
        
        # Extract the route points and distance/duration
        if directions_result:
            leg = directions_result[0]['legs'][0]
            route_points = leg['steps']
            coordinates = []
            
            for step in route_points:
                # Decode the polyline to get latitude/longitude
                polyline = step['polyline']['points']
                points = googlemaps.convert.decode_polyline(polyline)
                coordinates.extend([(point['lat'], point['lng']) for point in points])
                
            all_route_points.extend(coordinates)
            
            # Accumulate total distance and duration
            total_distance += leg['distance']['value']  # in meters
            total_duration += leg['duration']['value']  # in seconds
            
            origin = destination  # Update origin to the current destination for the next leg
            
    # Convert total distance from meters to miles
    total_distance_miles = total_distance * 0.000621371
    total_distance_min = total_duration/60
    
    return {
        'path': all_route_points,
        'total_distance': total_distance_miles,
        'total_duration': total_distance_min
    }



def get_distance_and_time(origin, destination):
    gmaps = googlemaps.Client(key=google_api_key)
    directions_result = gmaps.directions(origin, destination)
    if not directions_result:
        return None, None
    distance = directions_result[0]['legs'][0]['distance']['value']  # in meters
    duration = directions_result[0]['legs'][0]['duration']['value']  # in seconds
    return distance, duration

# Define the Graph Neural Network model for TSP

class SimpleGNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleGNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
# Fixing random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Ensure deterministic settings for CUDA
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Initialize the GNN model
input_dim = 2  # Each coordinate has 2 dimensions: latitude and longitude
output_dim = 2  # Output should also have 2 dimensions for coordinates
gnn_model = SimpleGNN(input_dim, output_dim)

def calculate_euclidean_distance(coords):
    """ Calculate pairwise Euclidean distances between all coordinates. """
    return cdist(coords, coords, metric='euclidean')

def tsp_optimized_route_gnn(origin, destinations, max_epochs=500, patience=10):
    coords = [origin[0]] + destinations  # Extract the tuple from the origin list
    coords = np.array(coords)
    
    # Prepare data for GNN
    gnn_input_tensor = torch.FloatTensor(coords)

    # Initialize model, optimizer, and loss function
    gnn_model = SimpleGNN(input_dim=2, output_dim=2)
    optimizer = optim.Adam(gnn_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    #optimizer = optim.Adam(gnn_model.parameters(), lr=0.01)
    #criterion = nn.MSELoss()

    # Calculate Euclidean distance heuristic
    distances = calculate_euclidean_distance(coords)

    # Use heuristic to get an initial guess for the path (Nearest Neighbor)
    initial_path = [0]  # Start with the origin
    for _ in range(len(coords) - 1):
        last_node = initial_path[-1]
        nearest_neighbor = np.argmin([distances[last_node][i] if i not in initial_path else np.inf for i in range(len(coords))])
        initial_path.append(nearest_neighbor)

    # Training loop with early stopping
    best_loss = float('inf')
    early_stop_counter = 0
    best_path = initial_path

    # Log losses for analysis
    loss_log = []

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        output = gnn_model(gnn_input_tensor)
        target_tensor = gnn_input_tensor[initial_path]  # Use heuristic ordering as the initial target
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()

        # Log the loss
        loss_log.append(loss.item())
        #print(f"Epoch {epoch + 1}/{max_epochs}, Loss: {loss.item()}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_path = output.detach().numpy()  # Save the best output
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Early stopping
        if early_stop_counter >= patience:
            break

    optimized_coords = best_path   
    optimized_indices = np.argsort(np.linalg.norm(optimized_coords - gnn_input_tensor.numpy(), axis=1))

    # Create the best path after GNN optimization
    best_path_optimized = [coords[i] for i in optimized_indices]

    optimized_distance = []
    optimized_duration = []
    total_distance = 0
    total_time = 0

    for i in range(len(best_path_optimized) - 1):
        origin = best_path_optimized[i]
        destination = best_path_optimized[i + 1]
        
        distance, time = get_distance_and_time(origin, destination)  # Define this function separately
        if distance is not None and time is not None:
            optimized_distance.append(distance * 0.000621371)  # Convert to miles
            optimized_duration.append(time / 60)  # Convert to minutes

    total_distance = sum(optimized_distance)
    total_time = sum(optimized_duration)

    return {
        'best_path': best_path_optimized,
        'total_distance': total_distance,
        'total_duration': total_time,
        'distances': optimized_distance,
        'durations': optimized_duration,
        'loss_log': loss_log
    }

#***************************************************************
def plot_most_optimized_route(google_api_key, origin_coords, destination_coords_list):
    destination_coords_sorted = sorted(destination_coords_list, key=lambda x: (x[0], x[1]))
    
    total_optimized_distance = 0
    total_optimized_duration = 0
    total_original_distance = 0
    total_original_duration = 0

    distances = []
    durations = []


    destination_number = 1
 
    original_route_details = get_route_from_google_maps(google_api_key, origin_coords, destination_coords_sorted)

    origin_coords_array = [(float(origin_coords[0]), float(origin_coords[1]))]
    destination_coords_array = [(float(lat), float(lon)) for lat, lon in destination_coords_list]

        
    if len(destination_coords_sorted) < 3:
        optimized_route_details = original_route_details
    else:
        optimized_route_details = tsp_optimized_route_gnn(origin_coords_array, destination_coords_array)

    # Convert best_path to float values for AntPath
    best_path = [(float(lat), float(lon)) for lat, lon in optimized_route_details['best_path']]
    optimized_route = get_route_from_google_maps(google_api_key, origin_coords, best_path)
    
    # Calculate total optimized distance and duration
    total_optimized_distance += round(optimized_route_details['total_distance'], 2)
    total_optimized_duration += round(optimized_route_details['total_duration'], 2) 
    total_original_distance += round(original_route_details['total_distance'], 2)  # Convert to miles
    total_original_duration += round(original_route_details['total_duration'], 2)  # Convert to minutes

    for distance in optimized_route_details['distances']:      
        distances.append(round(float(distance), 2))
    for duration in optimized_route_details['durations']:  
        durations.append(round(float(duration), 2))

#************************* PLot Map *************************        
    # Initialize the Folium map at the origin
    map_obj = folium.Map(location=origin_coords, zoom_start=13)
    # Add AntPath for the route
    AntPath(locations=optimized_route['path'], dash_array=[10, 10], delay=1000, color='red', pulse_color='orange').add_to(map_obj)
    
    for destination in best_path:
        folium.Marker(
            location=destination,
            popup=f'Destination {destination_number}: {destination}',
            icon=folium.Icon(color='blue')
        ).add_to(map_obj)
        destination_number += 1
        
    # Add markers for origin and destination
    folium.Marker(location=origin_coords, popup=f'Origin: {origin_coords}', icon=folium.Icon(color='green')).add_to(map_obj)

        
    # Save the map as an HTML file
    map_path = 'static/optimized_route_map.html'
    map_obj.save(map_path)
    
    # Return the relevant data, including the path to the map and route details
    return {
        'map_path': map_path,
        'best_path': best_path,
        'original_distance': total_original_distance,
        'original_duration': total_original_duration,
        'optimized_distance': total_optimized_distance,
        'optimized_duration': total_optimized_duration,
        'distances': distances,
        'durations': durations
    }

#**********************************   K-means clusters    *******************************************
def create_clusters(customer_coords, max_cluster_size=10):

    # If there are fewer than or equal to max_cluster_size customers, return as a single cluster
    if len(customer_coords) <= max_cluster_size:
        return [customer_coords]

    # Calculate the number of clusters needed
    num_clusters = len(customer_coords) // max_cluster_size

    # If less than 1 cluster would be created, set num_clusters to 1
    num_clusters = max(1, num_clusters)

    # Convert customer coordinates to a NumPy array for KMeans
    coords_array = np.array(customer_coords)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    labels = kmeans.fit_predict(coords_array)

    # Create clusters based on KMeans labels
    clusters = [[] for _ in range(num_clusters)]
    for idx, label in enumerate(labels):
        clusters[label].append(customer_coords[idx])

    # Ensure each cluster does not exceed max_cluster_size
    final_clusters = []
    for cluster in clusters:
        for i in range(0, len(cluster), max_cluster_size):
            final_clusters.append(cluster[i:i + max_cluster_size])

    print('Clusters:', final_clusters)
    return final_clusters

#*****************************************************************************************************

@app.route('/select_city', methods=['POST'])
def select_city():
    selected_city = request.form.get('selected_city')
    return redirect(url_for('index', selected_city=selected_city))

def get_cities():
    conn = get_snowflake_connection()
    try:
        query = "SELECT DISTINCT FAC_CITY FROM CUST_DATA_TB ORDER BY FAC_CITY"
        df = pd.read_sql(query, conn)
        cities = df['city'].tolist()
        cursor.close()
    except Exception as e:
        print(f"Error fetching cities: {e}")
        cities = []
    finally:
        conn.close()
    return cities


@app.route('/', methods=['GET'])
def index():
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT FAC_CITY FROM CUST_DATA_TB")
    cities = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()

    return render_template(
        'index_route.html',
        cities=cities,
        map_available=False
    )

@app.route('/', methods=['POST'])
def submit():
    selected_city = request.form.get('selected_city')
    selected_cluster = request.form.get('selected_cluster')
    selected_cluster_index = None  # Initialize to None or an appropriate default value


    conn = get_snowflake_connection()
    try:
        cities = get_cities()  
        print(f"Cities retrieved: {cities}")

        # If a city is selected, fetch facility and customer details
        if not selected_city:
            return render_template(
                'index_route.html', 
                cities=cities, 
                error="City selection is required.",
                map_available=False
            )
        else:
            fac_details = get_facility_details(selected_city)
            cust_details = get_addresses_for_city(selected_city)

            if not fac_details or not cust_details:
                return render_template(
                    'index_route.html', 
                    cities=cities, 
                    selected_city=selected_city, 
                    error="No data found for the selected city.",
                    map_available=False
                )

            origin_coords = (fac_details.get('FAC_LATITUDE'), fac_details.get('FAC_LONGITUDE'))
            # Cluster customer addresses into groups of 10
            customer_coords = [(float(addr['CUST_LATITUDE']), float(addr['CUST_LONGITUDE'])) for addr in cust_details]

            clusters = create_clusters(customer_coords)

            # Create a list for the dropdown in the format "city_name(cluster_number)"
            cluster_options = [f"{selected_city}({i+1})" for i in range(len(clusters))]

            # Select the specified cluster from the dropdown
            if selected_cluster_index and selected_cluster_index.isdigit():
                selected_cluster_index = int(selected_cluster_index)
                selected_cluster = clusters[selected_cluster_index] if clusters else []
            else:
                selected_cluster = clusters[0] if clusters else []
        
        destination_coords_list = list(set(selected_cluster))

        optimized_route_details = plot_most_optimized_route(google_api_key, origin_coords, destination_coords_list)

        # Handle None values before rendering
        distance = optimized_route_details.get('distance')
        if distance is not None:
            distance = round(distance, 2)

        # Rearrange cust_details based on best_path
        rearranged_cust_details = []
        for coord in optimized_route_details['best_path']:
            if coord in selected_cluster:
                index = selected_cluster.index(coord)
                rearranged_cust_details.append(cust_details[index])

        # Other details
        optimized_route_details = {
            'map_path': optimized_route_details.get('map_path', ''),
            'best_path': optimized_route_details.get('best_path', []),
            'distances': optimized_route_details.get('distances', []),
            'durations': optimized_route_details.get('durations', []),
            'original_distance': optimized_route_details.get('original_distance', None),
            'optimized_distance': optimized_route_details.get('optimized_distance', None),
            'original_duration': optimized_route_details.get('original_duration', None),
            'optimized_duration': optimized_route_details.get('optimized_duration', None)
        }
        
        if optimized_route_details['map_path']:
            return render_template(
                'index_route.html', 
                map_available=True, 
                cities=cities,
                selected_city=selected_city,
                name=fac_details.get('FAC_NAME'),
                address=fac_details.get('FAC_ADDRESS'),
                city=fac_details.get('FAC_CITY'),
                state=fac_details.get('FAC_STATE'),
                latitude=fac_details.get('FAC_LATITUDE'),
                longitude=fac_details.get('FAC_LONGITUDE'),
                cust_details=[
                    (addr['CUST_ADDRESS'], addr['CUST_CITY'], addr['CUST_STATE'], addr['CUST_LATITUDE'], addr['CUST_LONGITUDE']) 
                    for addr in rearranged_cust_details],
                customer_coords=destination_coords_list,
                origin_coords=origin_coords,
                best_path=optimized_route_details['best_path'],
                map_path=optimized_route_details['map_path'],
                distances=optimized_route_details['distances'],
                durations=optimized_route_details['durations'],
                original_distance=optimized_route_details['original_distance'],
                optimized_distance=optimized_route_details['optimized_distance'],
                original_duration=optimized_route_details['original_duration'],
                optimized_duration=optimized_route_details['optimized_duration']
            )
        else:
            print("Failed to plot the optimized route.")
            return render_template(
                'index_route.html', 
                map_available=False,
                cities=cities,
                selected_city=selected_city
            )
    finally:
        conn.close()
     

@app.route('/display_map')
def display_map():
    return send_file('map.html')


if __name__ == '__main__':
    app.run(debug=True)
