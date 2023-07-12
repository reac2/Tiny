import ee
import glob
import os
import geemap.foliumap as emap
import rasterio
import logging
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.colors import LinearSegmentedColormap
logging.getLogger('rasterio').setLevel(logging.ERROR)

def mask_from_bitmask(bitmask, mask_type):
    '''
    Converts an earth engine QA bitmask to a boolean array of ``mask_type`` pixels.
    Bitmask array conversion from https://stackoverflow.com/questions/22227595/convert-integer-to-binary-array-with-suitable-padding

    Args:
        bitmask (np.ndarray): ``shape(pix_x, pix_y)``
        mask_type (str): String specifying kind of mask to return. Can be ``water``, ``cloud``, or ``shadow``.

    Returns:
        Boolean mask with shape ``shape(pix_x, pix_y)`` indicating pixels of ``mask_type.
    '''

    idx_dict = {
        "water" : 8,
        "cloud" : 12,
        "shadow" : 11,
    }

    # number of bits to convert bitmask to
    m = 16
    # Function to convert an integer to a string binary representation
    to_str_func = np.vectorize(lambda x: np.binary_repr(x).zfill(m))
    # Calculte binary representations
    strs = to_str_func(bitmask)
    # Create empty array for the bitmask
    bitmask_bits = np.zeros(list(bitmask.shape) + [m], dtype=np.int8)
    # Iterate over all m  bits
    for bit_ix in range(0, m):
        # Get the bits
        fetch_bit_func = np.vectorize(lambda x: x[bit_ix] == '1')
        # Store the bits
        bitmask_bits[:, :, bit_ix] = fetch_bit_func(strs).astype("int8")

    # The water bitmask is stored in bit 7 (index 15-7=8).
    bool_bitmask = bitmask_bits[:, :, idx_dict[mask_type]] == 1

    return bool_bitmask

def get_water_depth(rbgnir):
    rbgnir = (rbgnir.astype(np.float32))
    water_mask = mask_from_bitmask(rbgnir[:, :, 3].astype(np.int64), 'water')
    depth = np.zeros_like(rbgnir[:, :, 0])
    depth[water_mask] = ((np.log(rbgnir[:, :, 0][water_mask])/np.log(rbgnir[:, :, 1][water_mask])))
    return depth, water_mask

def scale_im(reader):
    red = reader.read(3)
    green = reader.read(2)
    blue = reader.read(1)

    scale = lambda x : (x*0.0000275) - 0.2

    return np.dstack([scale(red), scale(green), scale(blue)]) * 3


def create_color_map(hex_list):
    num_colors = len(hex_list)
    color_positions = np.linspace(0, 1, num_colors)
    color_map_dict = {'red': [], 'green': [], 'blue': []}

    for color_index, hex_color in enumerate(hex_list):
        rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        color_map_dict['red'].append((color_positions[color_index], rgb_color[0] / 255, rgb_color[0] / 255))
        color_map_dict['green'].append((color_positions[color_index], rgb_color[1] / 255, rgb_color[1] / 255))
        color_map_dict['blue'].append((color_positions[color_index], rgb_color[2] / 255, rgb_color[2] / 255))

    color_map = LinearSegmentedColormap('custom_color_map', color_map_dict)
    return color_map

def calculate_square_corners(pointlat, pointlong, pad=0.1):
    toplat = pointlat + pad
    toplong = pointlong + pad
    botlat = pointlat - pad
    botlong = pointlong - pad
    return toplat, toplong, botlat, botlong


def random_date_tuple():
    start_date = datetime(2013, 1, 1)
    end_date = datetime(2022, 12, 31)
    month_interval = timedelta(days=30)

    start_timestamp = start_date.timestamp()
    end_timestamp = end_date.timestamp()

    random_timestamp = random.uniform(start_timestamp, end_timestamp)
    random_date = datetime.fromtimestamp(random_timestamp)

    start_of_month = datetime(random_date.year, random_date.month, 1)
    end_of_month = start_of_month + month_interval

    start_date_str = start_of_month.strftime('%Y-%m-%d')
    end_date_str = end_of_month.strftime('%Y-%m-%d')

    return start_date_str, end_date_str

class MakeTrainingData:
    '''
    A class to make some nice training data for the depth of the sea

    '''
    def __init__(
        self,
        path,
        toplat,
        toplong,
        botlat,
        botlong,
        start_year,
        end_year,
        SR_PROD='LANDSAT/LC08/C02/T1_L2',
        RAW_PROD='LANDSAT/LC08/C02/T1',
        SR_BANDS=['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'QA_PIXEL'],
        RAW_BANDS=['B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
        MAX_CLOUD_COVER=10,
        ):
        '''Initialize the MakeTrainingData class.

        Args:
        - path (str): Path to the directory where the data will be stored.
        - toplat (float): Latitude of the top left corner of the region of interest.
        - toplong (float): Longitude of the top left corner of the region of interest.
        - botlat (float): Latitude of the bottom right corner of the region of interest.
        - botlong (float): Longitude of the bottom right corner of the region of interest.
        - start_year (int): Starting year for the Landsat imagery.
        - end_year (int): Ending year for the Landsat imagery.
        - SR_PROD (str, optional): Landsat surface reflectance product. Defaults to 'LANDSAT/LC08/C02/T1_L2'.
        - RAW_PROD (str, optional): Landsat raw product. Defaults to 'LANDSAT/LC08/C02/T1'.
        - SR_BANDS (list, optional): Bands to select from the surface reflectance product. Defaults to ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'QA_PIXEL'].
        - RAW_BANDS (list, optional): Bands to select from the raw product. Defaults to ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'].
        - MAX_CLOUD_COVER (int, optional): Maximum cloud cover percentage. Defaults to 10.
        '''

        # Set up region of interest
        region_of_interest = [toplat, toplong, botlat, botlong]
        self.region =  ee.Geometry.Rectangle(region_of_interest)

        # Set up other params
        self.SR_PROD = SR_PROD
        self.RAW_PROD = RAW_PROD
        self.SR_BANDS = SR_BANDS
        self.RAW_BANDS = RAW_BANDS
        self.MAX_CLOUD_COVER = MAX_CLOUD_COVER
        self.SR_PATH = path + '/SR'
        self.RAW_PATH = path + '/RAW'

        # Make paths if not present
        if not os.path.isdir(self.SR_PATH):
            os.mkdir(self.SR_PATH)
        if not os.path.isdir(self.RAW_PATH):
            os.mkdir(self.RAW_PATH)


        # Make date list
        years = [str(i) for i in range(start_year,end_year)]
        months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
        date_list = [f"{year}-{month}-01" for year in years for month in months]
        date_list.append("2019-01-01")
        self.dates = [(date_list[i], date_list[i+1]) for i in range(len(date_list)-1)]

    def download_data(self):
        '''Download the Landsat imagery data for the specified region of interest and time range.'''
        # Define max water count and best file
        water_max = 0
        self.best_file = ''

        # Iterate through the months
        for start_date, end_date in self.dates:

            # Define paths for each month
            SR_OUT_PATH = f'{self.SR_PATH}/{start_date}'
            if not os.path.isdir(SR_OUT_PATH):
                os.mkdir(SR_OUT_PATH)

            RAW_OUT_PATH = f'{self.RAW_PATH}/{start_date}'
            if not os.path.isdir(RAW_OUT_PATH):
                os.mkdir(RAW_OUT_PATH)


            # Convert the dates to a form earth engine likes
            start_date = ee.Date(start_date)
            end_date = ee.Date(end_date).advance(-1, "day")

            # Filter input collections by desired date range, region and cloud coverage.
            criteria  = ee.Filter.And(
                ee.Filter.geometry(self.region),
                ee.Filter.date(start_date, end_date)
            )

            # Get the surface reflectance collection
            SR = ee.ImageCollection(self.SR_PROD) \
                            .filter(criteria) \
                            .filter(ee.Filter.lt('CLOUD_COVER', self.MAX_CLOUD_COVER)) \
                            .select(self.SR_BANDS)

            # Get the RAW collection
            RAW = ee.ImageCollection(self.RAW_PROD) \
                .filter(criteria) \
                .filter(ee.Filter.lt('CLOUD_COVER', self.MAX_CLOUD_COVER)) \
                .select(self.RAW_BANDS)

            # Export the surface reflectance collection to the SR_OUT_PATH directory
            emap.ee_export_image_collection(
                    SR,
                    SR_OUT_PATH,
                    crs='EPSG:4326',
                    scale=30,
                    region=self.region
                )

            # Export the RAW collection to the RAW_OUT_PATH directory
            emap.ee_export_image_collection(
                    RAW,
                    RAW_OUT_PATH,
                    crs='EPSG:4326',
                    scale=30,
                    region=self.region
                )

            # Loop through the files in the surface reflectance folder
            for file in glob.glob(SR_OUT_PATH+'/*'):
                # Open the file
                reader = rasterio.open(file)
                # Get the QA pixels
                qa_pix = reader.read(5)
                # Get the water mask
                water_mask = mask_from_bitmask(qa_pix, 'water')
                # Is the number of water pixels bigger than the current max?
                if water_max < water_mask.sum():
                    # If so, update the water max and the best file
                    water_max = water_mask.sum()
                    self.best_file = file

    def get_training_data(self, LAND_FILL_VALUE=0):
        '''Process the downloaded Landsat imagery data and extract training data for sea depth estimation.

        Args:
        - LAND_FILL_VALUE (int, optional): Value to fill for land pixels. Defaults to -1.

        Returns:
        - X (ndarray): Input training data.
        - y (ndarray): Target training data.
        '''

        # BUGGY! This throws an error every time hence the try except, but the download works.
        # Download data
        try:
            self.download_data()
        except:
            pass

        # Define readers for the best file
        SR_reader = rasterio.open(self.best_file)
        RAW_reader = rasterio.open(self.best_file.replace('SR', 'RAW'))

        # Get the best image
        self.best_im = scale_im(SR_reader)

        # Extract relevant bands from surface reflectance
        bgnir = np.dstack([
            SR_reader.read(1),
            SR_reader.read(2),
            SR_reader.read(4),
            SR_reader.read(5)
        ]).astype(np.float32)

        # Get the depth
        depth, mask = get_water_depth(bgnir)

        # Annoying infs and nans handling (not sure if needed now)
        ninf = ~(depth == -np.inf)
        nnan = ~np.isnan(depth)
        nall = np.logical_and(nnan, ninf)
        self.full_mask = np.logical_and(nall, mask)
        # Get the water depth
        self.full_depth = np.full_like(depth, LAND_FILL_VALUE)
        self.full_depth[self.full_mask] = depth[self.full_mask]

        # Get the raw data
        X = np.dstack([
            RAW_reader.read(1),
            RAW_reader.read(2),
            RAW_reader.read(3),
            RAW_reader.read(4),
            RAW_reader.read(5),
            RAW_reader.read(6),
        ])

        # Reshape X from (pix_x, pix_y, bands) to (num_pix, bands)
        X = X.reshape(X.shape[0]*X.shape[1], X.shape[2])

        # Reshape the depth to get y
        y =  self.full_depth.reshape(-1)

        return X, y


    def plot_depth(self):
        '''Plot the sea depth map for the specified region of interest based on the processed Landsat imagery data.'''
        # Funky colors
        cmap_list = ['00ff44', '4FA0A2', '404E8C', '291C2E', ]
        cmap = create_color_map(cmap_list)
        # janky
        depth = self.full_depth
        water_mask = self.full_mask
        best_file = self.best_file

        # Janky scaling
        scale = lambda x : (x - x[~np.isnan(x)].min())/(x[~np.isnan(x)].max() - x[~np.isnan(x)].min())
        depth[np.logical_and(depth!=-np.inf, water_mask)] = scale(depth[np.logical_and(depth!=-np.inf, water_mask)])
        depth[~np.logical_and(depth!=-np.inf, water_mask)]  = depth[~np.logical_and(depth!=-np.inf, water_mask)].min()

        # Mask the depth array
        depth = np.ma.masked_array(depth, ~water_mask, fill_value=np.nan)
        self.im = scale_im(rasterio.open(best_file))
        #im = np.ma.masked_array(im, np.dstack([water_mask, water_mask, water_mask]))

        # Set up figure
        plt.figure(dpi=300)

        # Plot image
        plt.imshow(self.im)

        # Plot the depth where the depth is
        d = plt.imshow(depth, cmap=cmap)

        # Add a colorbar
        plt.colorbar(d, fraction=0.036, pad=0.04)

        # Show the plot
        plt.show()



class MakeWaterData:
    '''
    A class to make some nice training data for the presence of water

    '''
    def __init__(
        self,
        path_to_ims,
        toplat,
        toplong,
        botlat,
        botlong,
        date_range,
        SR_PROD='LANDSAT/LC08/C02/T1_L2',
        RAW_PROD='LANDSAT/LC08/C02/T1',
        SR_BANDS=['QA_PIXEL'],
        RAW_BANDS=['B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
        MAX_CLOUD_COVER=50,
        ):
        '''Initialize the MakeTrainingData class.

        Args:
        - path (str): Path to the directory where the data will be stored.
        - toplat (float): Latitude of the top left corner of the region of interest.
        - toplong (float): Longitude of the top left corner of the region of interest.
        - botlat (float): Latitude of the bottom right corner of the region of interest.
        - botlong (float): Longitude of the bottom right corner of the region of interest.
        - start_year (int): Starting year for the Landsat imagery.
        - end_year (int): Ending year for the Landsat imagery.
        - SR_PROD (str, optional): Landsat surface reflectance product. Defaults to 'LANDSAT/LC08/C02/T1_L2'.
        - RAW_PROD (str, optional): Landsat raw product. Defaults to 'LANDSAT/LC08/C02/T1'.
        - SR_BANDS (list, optional): Bands to select from the surface reflectance product. Defaults to ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'QA_PIXEL'].
        - RAW_BANDS (list, optional): Bands to select from the raw product. Defaults to ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'].
        - MAX_CLOUD_COVER (int, optional): Maximum cloud cover percentage. Defaults to 10.
        '''

        # Set up region of interest
        region_of_interest = [toplat, toplong, botlat, botlong]
        self.region =  ee.Geometry.Rectangle(region_of_interest)

        # Set up other params
        self.SR_PROD = SR_PROD
        self.RAW_PROD = RAW_PROD
        self.SR_BANDS = SR_BANDS
        self.RAW_BANDS = RAW_BANDS
        self.MAX_CLOUD_COVER = MAX_CLOUD_COVER
        self.SR_PATH = path_to_ims + '/SR'
        self.RAW_PATH = path_to_ims + '/RAW'
        self.date_range = date_range

        # Make paths if not present
        if not os.path.isdir(self.SR_PATH):
            os.mkdir(self.SR_PATH)
        if not os.path.isdir(self.RAW_PATH):
            os.mkdir(self.RAW_PATH)


        # Make date list
        # years = [str(i) for i in range(start_year,end_year)]
        # months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
        # date_list = [f"{year}-{month}-01" for year in years for month in months]
        # date_list.append("2019-01-01")
        # self.dates = [(date_list[i], date_list[i+1]) for i in range(len(date_list)-1)]

    def download_data(self):
        '''Download the Landsat imagery data for the specified region of interest and time range.'''
        # Define max water count and best file
        X = []
        y = []

        # Iterate through the months
        start_date, end_date = self.date_range

        # Define paths for each month
        SR_OUT_PATH = f'{self.SR_PATH}/{start_date}'
        if not os.path.isdir(SR_OUT_PATH):
            os.mkdir(SR_OUT_PATH)

        RAW_OUT_PATH = f'{self.RAW_PATH}/{start_date}'
        if not os.path.isdir(RAW_OUT_PATH):
            os.mkdir(RAW_OUT_PATH)


        # Convert the dates to a form earth engine likes
        start_date = ee.Date(start_date)
        end_date = ee.Date(end_date).advance(-1, "day")

        # Filter input collections by desired date range, region and cloud coverage.
        criteria  = ee.Filter.And(
            ee.Filter.geometry(self.region),
            ee.Filter.date(start_date, end_date)
        )

        # Get the surface reflectance collection
        SR = ee.ImageCollection(self.SR_PROD) \
                        .filter(criteria) \
                        .filter(ee.Filter.lt('CLOUD_COVER', self.MAX_CLOUD_COVER)) \
                        .select(self.SR_BANDS)

        # Get the RAW collection
        RAW = ee.ImageCollection(self.RAW_PROD) \
            .filter(criteria) \
            .filter(ee.Filter.lt('CLOUD_COVER', self.MAX_CLOUD_COVER)) \
            .select(self.RAW_BANDS)

        # Export the surface reflectance collection to the SR_OUT_PATH directory
        emap.ee_export_image_collection(
                SR,
                SR_OUT_PATH,
                crs='EPSG:4326',
                scale=30,
                region=self.region
            )

        # Export the RAW collection to the RAW_OUT_PATH directory
        emap.ee_export_image_collection(
                RAW,
                RAW_OUT_PATH,
                crs='EPSG:4326',
                scale=30,
                region=self.region
            )

        # Loop through the files in the surface reflectance folder
        for file in glob.glob(SR_OUT_PATH+'/*'):
            print(file)
            # Open the file
            reader = rasterio.open(file)
            # Get the QA pixels
            qa_pix = reader.read(1)
            # Get the water mask
            water_mask = mask_from_bitmask(qa_pix, 'water').reshape(-1)
            # Get the raw file
            RAW_reader  = rasterio.open(file.replace('SR', 'RAW'))

            # Get the raw image
            raw_im = np.dstack([
                RAW_reader.read(1),
                RAW_reader.read(2),
                RAW_reader.read(3),
                RAW_reader.read(4),
                RAW_reader.read(5),
                RAW_reader.read(6),
            ])
            # add to training data
            X.extend(raw_im.reshape(water_mask.shape[0], 6))
            y.extend(water_mask)

                # # Is the number of water pixels bigger than the current max?
                # if water_max < water_mask.sum():
                #     # If so, update the water max and the best file
                #     water_max = water_mask.sum()
                #     self.best_file = file
        return np.array(X), np.array(y)


    def plot_depth(self):
        '''Plot the sea depth map for the specified region of interest based on the processed Landsat imagery data.'''
        # Funky colors
        cmap_list = ['00ff44', '4FA0A2', '404E8C', '291C2E', ]
        cmap = create_color_map(cmap_list)
        # janky
        depth = self.full_depth
        water_mask = self.full_mask
        best_file = self.best_file

        # Janky scaling
        scale = lambda x : (x - x[~np.isnan(x)].min())/(x[~np.isnan(x)].max() - x[~np.isnan(x)].min())
        depth[np.logical_and(depth!=-np.inf, water_mask)] = scale(depth[np.logical_and(depth!=-np.inf, water_mask)])
        depth[~np.logical_and(depth!=-np.inf, water_mask)]  = depth[~np.logical_and(depth!=-np.inf, water_mask)].min()

        # Mask the depth array
        depth = np.ma.masked_array(depth, ~water_mask, fill_value=np.nan)
        self.im = scale_im(rasterio.open(best_file))
        #im = np.ma.masked_array(im, np.dstack([water_mask, water_mask, water_mask]))

        # Set up figure
        plt.figure(dpi=300)

        # Plot image
        plt.imshow(self.im)

        # Plot the depth where the depth is
        d = plt.imshow(depth, cmap=cmap)

        # Add a colorbar
        plt.colorbar(d, fraction=0.036, pad=0.04)

        # Show the plot
        plt.show()




