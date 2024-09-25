from flask import Flask, request, render_template
import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import requests
from geopy.distance import great_circle
from flask import Blueprint

app = Flask(__name__)

google_api_key = 'google_api_key'
cust_tb = 'CUST_DATA_TB'

route_blueprint = Blueprint('route_blueprint', __name__, url_prefix='/Pathsync/route_optimization')

def get_snowflake_connection():
    return snowflake.connector.connect(
        user='user',
        password='passowrd',
        account='account',
        warehouse='WH',
        database='DB',
        schema='SC'
    )

def check_snf_conn(conn):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT CURRENT_VERSION()")
            data = cur.fetchone()
            print("Snowflake Version:", data[0])
    finally:
        conn.close()

def get_filtered_facilities():
    conn = get_snowflake_connection()
    query = "SELECT * FROM FAC_LOC_TB"
    df = pd.read_sql(query, conn)
    df.columns = [col.upper() for col in df.columns]  
    return df

def calculate_distance(coord1, coord2):
    return great_circle(coord1, coord2).miles

      
def get_nearest_facilities(customer_coords):
    filtered_facilities = get_filtered_facilities()

    coordinates = list(zip(filtered_facilities['LATITUDE'], filtered_facilities['LONGITUDE']))
    
    nearest_facility_details = []

    distances = [calculate_distance(customer_coords, coord) for coord in coordinates]

    filtered_facilities['DISTANCE'] = distances

    nearest_facility = filtered_facilities.loc[filtered_facilities['DISTANCE'].idxmin()]

    return nearest_facility
    

def get_address_details(customer_address, google_api_key):
    base_url = 'https://maps.googleapis.com/maps/api/geocode/json'
    params = {'address': customer_address, 'key': google_api_key}
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        geocode_result = data.get('results', [])
        if data['status'] == 'OK' and geocode_result:
            location = geocode_result[0]['geometry']['location']
            lat, lng = location['lat'], location['lng']
        
            address_components = geocode_result[0]['address_components']
            city, state, zip_code = None, None, None
        
            for component in address_components:
                if 'locality' in component['types']:
                    city = component['long_name']
                if 'administrative_area_level_1' in component['types']:
                    state = component['short_name']
                if 'postal_code' in component['types']:
                    zip_code = component['long_name']
        
            customer_coords = (lat, lng)
            nearest_facility = get_nearest_facilities(customer_coords)
            
            if nearest_facility is not None:

                return {
                    'CUST_ADDRESS': customer_address,
                    'CUST_CITY': city,
                    'CUST_STATE': state,
                    'CUST_ZIP': zip_code,
                    'CUST_LATITUDE': lat,
                    'CUST_LONGITUDE': lng,
                    'FAC_NAME': nearest_facility.get('NAME'),
                    'FAC_ADDRESS': nearest_facility.get('ADDRESS'),
                    'FAC_CITY': nearest_facility.get('CITY'),
                    'FAC_STATE': nearest_facility.get('STATE'),
                    'FAC_ZIP': nearest_facility.get('ZIP'),
                    'FAC_LATITUDE': nearest_facility.get('LATITUDE'),
                    'FAC_LONGITUDE': nearest_facility.get('LONGITUDE'),
                    'FAC_DISTANCE': nearest_facility.get('DISTANCE')
                }
            else:
                return None

        else:
            print(f"API error: {data['status']}")
            if 'error_message' in data:
                print(f"Error message: {data['error_message']}")
    else:
        print(f"HTTP request error: {response.status_code}")
        
    return None

    
def create_and_load_cust_table(cust_col, cust_data, cust_tb, unique_cols):
    if not isinstance(cust_data, pd.DataFrame):
        cust_data = pd.DataFrame(cust_data)
    
    conn = get_snowflake_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SHOW TABLES LIKE '{cust_tb}'")
            table_exists = cur.fetchone()

            if not table_exists:
                column_definitions = ', '.join([f'"{col.upper()}" STRING' for col in cust_col])
                unique_constraint = ', '.join([f'"{col.upper()}"' for col in unique_cols])
                create_table_sql = f"""
                CREATE OR REPLACE TABLE {cust_tb} (
                    {column_definitions},
                    CONSTRAINT unique_constraint UNIQUE ({unique_constraint})
                );
                """
                cur.execute(create_table_sql)
                message = f"Table {cust_tb} created successfully."
            else:
                message = f"Table {cust_tb} already exists."

            cust_data = cust_data.drop_duplicates(subset=unique_cols)
            existing_data_query = f"SELECT {', '.join([col.upper() for col in unique_cols])} FROM {cust_tb}"
            cur.execute(existing_data_query)
            existing_data = cur.fetchall()

            if existing_data:
                existing_df = pd.DataFrame(existing_data, columns=[col.upper() for col in unique_cols])
                cust_data = cust_data[~cust_data[unique_cols].apply(tuple, 1).isin(existing_df.apply(tuple, 1))]

            if not cust_data.empty:
                success, num_chunks, num_rows, num_cols = write_pandas(
                    conn,
                    cust_data,
                    cust_tb
                )
                message += f" Data loaded successfully: {num_rows} rows."
            else:
                message += " No new data to load."

    except Exception as e:
        message = f"Error: {e}"
    finally:
        conn.close()
    return message

@app.route('/')
def index_cust():
    return render_template('index_cust.html', map_available=False)

@app.route('/', methods=['POST'])
def submit():
    conn = get_snowflake_connection()
    check_snf_conn(conn)

    addresses_inp = {
        'address1': request.form.get('address1'),
        'address2': request.form.get('address2'),
        'address3': request.form.get('address3'),
        'address4': request.form.get('address4'),
        'address5': request.form.get('address5')
    }

    non_null_addresses = {k: v for k, v in addresses_inp.items() if v}

    df = pd.DataFrame(list(non_null_addresses.items()), columns=['Address Label', 'Address'])
    addresses = df['Address'].tolist()

    customer_details = []
    for addr in addresses:
        details = get_address_details(addr, google_api_key)
        if details:
            customer_details.append(details)

    unique_cols = ['CUST_LATITUDE', 'CUST_LONGITUDE']

    if customer_details:
        cust_col = pd.DataFrame(customer_details).columns
        message = create_and_load_cust_table(cust_col, customer_details, cust_tb, unique_cols)
    else:
        message = "No valid addresses provided."

    return render_template('index_cust.html', message=message)
    
app.register_blueprint(route_blueprint)

if __name__ == '__main__':
    app.run(debug=True)
