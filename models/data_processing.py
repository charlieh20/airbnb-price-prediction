import numpy as np
import pandas as pd

def min_max_normalization(data):
    return (data - data.min()) / (data.max() - data.min())

def log_normalization(data):
    log_data = np.log2(np.array(data))
    return min_max_normalization(log_data)

def count_normalization(data):
    counts = [len(x) for x in data]
    return min_max_normalization(np.array(counts))

def one_hot_encoding(data):
    unique = list(set(data))
    encoding_map = {}
    for idx, element in enumerate(unique):
        encoding_map[element] = idx
    return [encoding_map[x] for x in data]

def date_normalization(data):
    dates = pd.to_datetime(data)
    recent = dates.max()
    return min_max_normalization(np.array((recent - dates).dt.days))

def preprocessing(df, features):
    for feat in features:
        # Handle different ways of processing each feature
        if feat in ['amenities', 'host_verifications']:
            # Count normalization for 'amenities'
            df[feat] = count_normalization(df[feat])
        elif feat in ['accommodates']:
            # Log normalization for 'accommodates'
            df[feat] = log_normalization(df[feat])
        elif feat in ['host_is_superhost', 'host_has_profile_pic', 'room_type',
                      'host_identity_verified', 'has_availability', 
                      'instant_bookable', 'neighbourhood_cleansed', 
                      'neighbourhood_group_cleansed', 'property_type']:
            # Use a one-hot encoding for 'host_is_superhost'
            df[feat] = one_hot_encoding(df[feat])
        elif feat == 'host_since':
            # Normalize according to number of days from most recent date for 'host_since'
            df[feat] = date_normalization(df[feat])
        elif feat != 'id':
            # Min-Max normalization for eveything else
            df[feat] = min_max_normalization(df[feat])

    # Return the desired columns
    return df[features]