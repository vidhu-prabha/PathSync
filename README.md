
# PathSync: Intelligent Route Optimization using Geospatial Analysis, ML, Graph Neural Network 

The primary objective of **PathSync** is to develop a **Graph Neural Network (GNN) model** to optimize supply chain routing by effectively predicting the most efficient routes for deliveries. This model aims to enhance delivery speed and accuracy while minimizing costs and environmental impact through data-driven decision-making. Specifically, the project aims to achieve the following goals: 

* **Develop a GNN-based Model:** Create a robust Graph Neural Network model to optimize routing in supply chain management, leveraging facility and customer location data. 

* **Integrate Real-time Data:** Incorporate real-time traffic and weather data to enhance the accuracy and reliability of route predictions. 

* **Develop interactive visual reports** using Python libraries such as Folium and Matplotlib to visualize route optimization results, including distance, time, and cost metrics. 

* **Provide data-driven forecasts** of route efficiency over various time periods (daily, weekly, monthly, yearly) and estimate their potential impact on operations.

* **Deliver Actionable Insights:** Provide actionable insights and recommendations for supply chain managers to improve operational efficiency and reduce costs through optimized routing solutions. 

* **Assess Scalability:** Test the model's scalability by applying it to various supply chain scenarios with differing complexities and sizes. 

* **Enhance Prediction Accuracy:** Aim to improve the accuracy of delivery time predictions by utilizing the inherent relationships between facilities and customer locations in the GNN framework. 

* **Optimize Resource Utilization:** Investigate methods to optimize resource allocation (e.g., vehicles and personnel) based on the optimized routes generated by the GNN model. 

* **Develop a User-friendly Interface:** Create an intuitive interface for stakeholders to visualize route optimizations and assess their implications on supply chain operations. 

* **Facilitate Decision-Making:** Support decision-making processes by providing comprehensive reports and visualizations that highlight the benefits of adopting GNN- based route optimization. 


## **Project Steps and Modules**

Structured list of steps involved in this project on supply chain route optimization, including the modules and functionalities, this structured approach ensures a comprehensive handling of each aspect of the project, from data acquisition to analysis and reporting.:

**Project Initialization**
* Set up the virtual environment.
* Install necessary libraries (e.g., Pandas, NumPy, Matplotlib, Geopy, Scikit-learn, TensorFlow).

**Data Acquisition**
* Load UPS facility and customer data into Snowflake database
* Retrieve customer address data from the Snowflake database.

**Geocoding**
* Use the Google Maps API to convert customer addresses into latitude and longitude coordinates.
* Store the geocoded coordinates in a DataFrame.

**Data Preprocessing**
* Cleab UPS facilty data before loading it into Snowflake databse.
* Clean and preprocess the customer address data.
* Validate the presence of necessary columns (e.g., Address, City, State, Latitude, Longitude).

**Clustering Customer Addresses**
* Implement KMeans clustering to group customer addresses based on proximity (e.g., clustering into groups of 10).
* Calculate centroids for each cluster.

**Route Optimization Logic**

  For each cluster of customer addresses:

* Retrieve nearest UPS facility coordinates based on the cluster centroid.

* Implement a Traveling Salesman Problem (TSP) solution using a Graph Neural Network (GNN) to determine the optimal route.

* Calculate distances and durations for the optimized route.

**Distance and Duration Estimation**

* Estimate original and optimized distances and durations for each cluster.

* Store these estimations in a structured format.

**Cost and Savings Calculation**

* Calculate fuel costs and savings based on distances, fuel efficiency, and fuel prices.

* Implement functions to estimate these values for various time periods (daily, weekly, monthly, yearly).

**Data Storage**

* Store the results (original distances, optimized distances, durations, fuel costs, and savings) into a DataFrame for further analysis.

**Visualization**

* Create visualization for optimized route map.

* Create visualizations to compare original vs. optimized distances, durations, and fuel costs using Matplotlib.

* Generate summary statistics plots, including comparisons of distances, durations, fuel costs, and savings.

**Results Analysis**

* Analyze the results based on visualizations and summary statistics.

* Generate insights on optimization effectiveness.

**Documentation and Reporting**

* Document the code with comments and docstrings.

* Prepare a report summarizing findings, methodologies, and visualizations.

**Deployment (Optional)**

* Consider deploying the Flask application for user interactions.

* Ensure the application integrates with the Snowflake database and provides route optimization services.

### **Modules Overview**

* **Data Loading Module:** Handles loading customer and UPS facility data from Snowflake.

* **Geocoding Module:** Manages the conversion of addresses to coordinates using the Google Maps API.

* **Clustering Module:** Implements KMeans clustering for customer addresses.

* **Route Optimization Module:** Includes logic for TSP solution using GNN and distance calculation.

* **Estimation Module:** Performs calculations for distances, durations, fuel costs, and savings.

* **Visualization Module:** Handles the generation of plots and visual comparisons of results.




