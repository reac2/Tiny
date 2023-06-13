import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
from tqdm import tqdm
import os

def prune_quantize_save_model(model, X, y, path_to_tflite_model, BATCH_SIZE=1024):
    """
    Prunes, quantizes, and fine-tunes a Keras neural network model and saves it as a TFLite model.

    Args:
        model (tf.keras.Model): The Keras neural network model to be pruned and quantized.
        X (numpy.ndarray): The input data for training the model.
        y (numpy.ndarray): The target data for training the model.
        path_to_tflite_model (str): The file path to save the resulting TFLite model.
        BATCH_SIZE (int): The batch size to be used in training.

    Returns:
        None
    """

    # Define pruning parameters
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=0.9, begin_step=0, frequency=1000)
        #target_sparsity: how sparse you want the model to be (this might be key to look into)
        #begin_step: what stage in the training would you like to start pruining (this could be good to change to allow a pattern to form first and then prune)
        #frequency: how often do you prune (this could again be a good one to change to prune more often or less often but more vigourously perhaps)
    }

    # Define callbacks for updating pruning step
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep()
        # allows pruning to take place
    ]

    # Prune the model
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    pruned_model = prune_low_magnitude(model, **pruning_params)

    # Use a smaller learning rate for fine-tuning
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=1e-5)

    # Compiling the pruned model 
    pruned_model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=opt,
        metrics=['accuracy']
    )

    # Fine-tune the pruned model
    pruned_model.fit(
        X,
        y,
        batch_size=BATCH_SIZE,
        epochs=3,
        validation_split=0.1,
        callbacks=callbacks
    )
    
    def representative_dataset():
        # Generator function for representative dataset used in quantization
        for i in range(300):
            yield [X[i].astype(np.float32)]

    stripped_pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)

    # Annotate the model for quantization-aware training
    quant_aware_annotate_model = tfmot.quantization.keras.quantize_annotate_model(
        stripped_pruned_model)

    # Apply quantization to the model
    pqat_model = tfmot.quantization.keras.quantize_apply(
        quant_aware_annotate_model,
        tfmot.experimental.combine.Default8BitPrunePreserveQuantizeScheme())

    pqat_model.compile(
        optimizer=opt,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    # Fine-tune the quantized model
    pqat_model.fit(
        X,
        y,
        batch_size=BATCH_SIZE,
        epochs=3,
        validation_split=0.1
    )

    # This might be where the error is - This shouldnt' look this weird but
    # it's a workaround for running a CNN. If we had a FF network then we can
    # save it as a tflite model directly but for CNNs we need to do this. Maybe
    # we don't even need a CNN. 
    batch_size = 1
    input_shape = pqat_model.inputs[0].shape.as_list()
    input_shape[0] = batch_size
    func = tf.function(pqat_model).get_concrete_function(
        tf.TensorSpec(input_shape, pqat_model.inputs[0].dtype))
    converter = tf.lite.TFLiteConverter.from_concrete_functions([func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32 
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    # Save the TFLite model to a file
    with open(path_to_tflite_model, 'wb') as f:
        f.write(tflite_model)

    return None



def save_keras_as_tflite_base(model, path_to_tflite_model):
    """
    Converts and saves a Keras model as a TFLite model.

    Args:
        model (tf.keras.Model): The Keras model to be converted.
        path_to_tflite_model (str): The file path to save the resulting TFLite model.

    Returns:
        None
    """

    # Convert the Keras model to TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TFLite model to a file
    with open(path_to_tflite_model, 'wb') as f:
        f.write(tflite_model)

    return None

def run_tflite_model(path_to_tflite_model, X_data):
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
        x = x.astype(np.float32).reshape(1, 28, 28)
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


def get_tflite_model_size(path):
    
    # Returns the file size of the model given the path to it

    return os.path.getsize(path)/float(2**20)


