
#%%
from geopy.geocoders import Nominatim
from pyproj import Transformer
import pandas as pd
from tqdm import tqdm

def transform_coordinates(x :float, y: float, transformer: Transformer):

    """Transforms coordinates from one coordinate system to another"""    

    longitude, latitude = transformer.transform(x, y)
    return longitude, latitude


def reverse_geocode(row: pd.Series , geolocator:Nominatim ,loc_dict:dict):
    """Reverse geocodes the coordinates to get the location name of the cluster center."""
    
    if row.index[0] == -1:
        return "Unknown"
    location = geolocator.reverse((row['latitude'], row['longitude']))
    if location is None:
        return "Unknown"
    # Try location types from a town level up to a country level
    address_levels = location.raw['address'].keys()
    if 'town' in address_levels:
        loc = location.raw['address']['town']
    elif 'municipality' in address_levels:
        loc = location.raw['address']['municipality']
    elif 'state' in address_levels:
        loc = location.raw['address']['state']
    elif 'country' in address_levels:
        loc = location.raw['address']['country']
    else:
        loc = "Unknown"

    if loc not in loc_dict:
        loc_dict[loc] = 1
  
    else:
        loc_dict[loc] += 1
        loc =loc + str(loc_dict[loc])
    
    return loc


def get_cluster_names(cluster_df: pd.DataFrame,geolocator = Nominatim(user_agent="map_clusters"),
                      transformer = Transformer.from_crs("epsg:25832", "epsg:4326", always_xy=True) ) -> pd.DataFrame:
    """Returns the location names of the cluster centers."""

    loc_dict = {}

    cluster_centers = clusters_df.groupby('cluster')[['x_coords','y_coords']].mean()
    cluster_centers[['longitude', 'latitude']] = cluster_centers.apply(
        lambda row: transform_coordinates(row['x_coords'], row['y_coords'], transformer),
        axis=1,
        result_type='expand'
    )

    cluster_centers['loc_names'] = cluster_centers.apply(lambda x : reverse_geocode(x,geolocator,loc_dict), axis=1)
     
    return cluster_centers


if __name__ == "__main__":
    clusters_df = pd.read_csv("data_files/cluster_df.csv",index_col=0) 
    geolocator = Nominatim(user_agent="map_clusters")
    transformer = Transformer.from_crs("epsg:25832", "epsg:4326", always_xy=True)
    cluster_centers = get_cluster_names(clusters_df,geolocator,transformer)

# %%
