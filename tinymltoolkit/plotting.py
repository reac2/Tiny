import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import rasterio
from tinymltoolkit.data_download import mask_from_bitmask, scale_im, get_water_depth

def getTFlitePredictions32FLOAT(path_to_tflite_model, X_data):
    """
    Runs inference using a TFLite model on the given input data.

    Args:
        path_to_tflite_model (str): The file path to the TFLite model.
        X_data (numpy.ndarray): The input data for inference.

    Returns:
        numpy.ndarray: The predictions generated by the TFLite model.
    """

    n_points = X_data.shape[0]

    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=path_to_tflite_model)
    interpreter.allocate_tensors()

    # Get the input and output details from the TFLite model
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    preds = []
    for i, x in tqdm(enumerate(X_data), total=n_points):
        # Preprocess the input data
        x = x.astype(np.float32)
        # Set the input tensor
        interpreter.resize_tensor_input(input_index, x.shape)
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_index, x)
        interpreter.invoke()
        # Get the output tensor
        output = interpreter.tensor(output_index)
        # Copy the output data to avoid reference to internal data
        z = output()[0].copy()
        preds.append(z)
    return np.array(preds)

def robust_normalize(arr):
    # Calculate the median and median absolute deviation (MAD)
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))

    # Calculate the modified z-score for each element
    z_scores = 0.6745 * (arr - median) / mad

    # Set a threshold for outliers (e.g., z-score greater than 3.5)
    threshold = 3.5

    # Replace outliers with the median value
    normalized_arr = np.where(np.abs(z_scores) > threshold, median, arr)

    # Normalize the array using the maximum absolute value
    max_abs_value = np.max(np.abs(normalized_arr))
    normalized_arr /= max_abs_value

    return normalized_arr

def plot_on_ax(ax, img, depth, water_mask, cmap='turbo_r'):
    # Mask the depth array
    depth = np.ma.masked_array(depth, ~water_mask, fill_value=np.nan)
    # Set up figure
    depth[water_mask] = robust_normalize(depth[water_mask])
    depth[water_mask] = (((depth[water_mask]) - (depth[water_mask].min()))/((depth[water_mask].max())-(depth[water_mask].min())))
    # Plot image
    ax.imshow(img)
    # Plot the depth where the depth is
    d = ax.imshow(depth, cmap=cmap, vmin=0, vmax=1)
    # Add a colorbar
    #plt.colorbar(d, fraction=0.026, pad=0.04)
    return d


def run_models(water_image, depth_image, water_model, depth_model):
    shape = water_image.shape[:2]

    
    water_mask_unrolled = np.array(water_model(water_image.reshape((np.product(shape), 6))))[:, 1]
    water_mask = water_mask_unrolled.reshape(shape) > 0.5
    depth = np.empty_like(water_mask, dtype=np.float32)
    depth[water_mask] = np.array(depth_model(depth_image[water_mask]))[:, 0]

    return water_mask, depth



def make_plot(
        path_to_raw,
        keras_waternet,
        keras_depthnet,
        path_to_tflite_waternet,
        path_to_tflite_depthnet,
        scale_water,
        scale_depth
    ):

    tflite_waternet = lambda x : getTFlitePredictions32FLOAT(path_to_tflite_waternet, x)
    tflite_depthnet = lambda x : getTFlitePredictions32FLOAT(path_to_tflite_depthnet, x)

    RAW_reader = rasterio.open(path_to_raw)
    SR_reader = rasterio.open(path_to_raw.replace('RAW', 'SR'))
    img = scale_im(SR_reader)
    raw_img = scale_im(RAW_reader)
    RAW_image = np.dstack([
                RAW_reader.read(1),
                RAW_reader.read(2),
                RAW_reader.read(3),
                RAW_reader.read(4),
                RAW_reader.read(5),
                RAW_reader.read(6),
            ])

    test_image_water = scale_water.transform(RAW_image)
    test_image_depth = scale_depth.transform(RAW_image)
    bgnir = np.dstack([
            SR_reader.read(1),
            SR_reader.read(2),
            SR_reader.read(4),
            SR_reader.read(5)
    ]).astype(np.float32)

    ground_truth_depth, ground_truth_mask = get_water_depth(bgnir)

    keras_water_mask, keras_water_depth = run_models(
        test_image_water,
        test_image_depth,
        keras_waternet,
        keras_depthnet,
        )
    
    quantized_water_mask, quantized_water_depth = run_models(
        test_image_water,
        test_image_depth,
        tflite_waternet,
        tflite_depthnet,
    )
    
    fig, axs = plt.subplots(2, 3, dpi=400, sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.1, hspace=-0.25)

    axs[0, 0].imshow(img)
    axs[0, 0].set_title('SR')
    axs[1, 0].imshow(raw_img)
    axs[1, 0].set_title('RAW')
    _=plot_on_ax(axs[0, 1], img, ground_truth_depth, ground_truth_mask)
    axs[0, 1].set_title('Ground Truth')
    _=plot_on_ax(axs[1,1], img, keras_water_depth, keras_water_mask)
    axs[1, 1].set_title('Keras')
    d=plot_on_ax(axs[0, 2], img, quantized_water_depth, quantized_water_mask)
    axs[0, 2].set_title('Quantized')
    _=plot_on_ax(axs[1, 2], img, (keras_water_depth - quantized_water_depth), quantized_water_mask, cmap='Greys')
    axs[1, 2].set_title('K-Q',)

    for i in axs:
        for j in i:
            j.set_xticks([])
            j.set_yticks([])

    fig.colorbar(d, ax=axs, shrink=0.6,label='Normalised Depth' )
    #plt.tight_layout()
    plt.show()
    
