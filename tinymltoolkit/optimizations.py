import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
from tqdm import tqdm



def setPrunedModelh5(modelToBePruned,X_train,y_train,sparsity,frequency,pathToSave):
    #define pruning function
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    #define pruning parameters
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(sparsity,0,-1,frequency),
        'block_size': (1,1),
        'block_pooling_type': 'AVG'
    }
    prunedModel = prune_low_magnitude(modelToBePruned,**pruning_params)
    #pruning requires a recompile of the model
    optimiser = tf.keras.optimizers.legacy.Adam(learning_rate=1e-5)
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics = ['accuracy']
    prunedModel.compile(optimizer=optimiser,loss=loss,metrics=metrics)
    #train the model for a few epochs to actually prune it
    logDirectory = 'pruningSummary'
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir = logDirectory)
    ]
    prunedModel.fit(X_train,y_train,batch_size = 64, epochs = 4, validation_split = 0.1,callbacks=callbacks)
    #strip off the pruning layers 
    prunedModel = tfmot.sparsity.keras.strip_pruning(prunedModel)
    tf.keras.saving.save_model(model=prunedModel,filepath=pathToSave,save_format="h5")


def set_quantized_model_float16(model_to_be_quantized,quantized_model_path,X_train):
    converter = tf.lite.TFLiteConverter.from_keras_model(model_to_be_quantized)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    def representative_dataset():
        # Generator function for representative dataset used in quantization
        for i in range(len(X_train)):
            yield [X_train[i].astype(np.float32)]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.float16]
    converter.inference_input_type = tf.float32  # or tf.uint8
    converter.inference_output_type = tf.float32  # or tf.uint8
    quantizedModelTFlite = converter.convert()
    with open(quantized_model_path, 'wb') as f:
        f.write(quantizedModelTFlite)




def setQuantizedModel(modelToBeQuantizedPath,quantizedModelPath,X_train):
    converter = tf.lite.TFLiteConverter.from_keras_model(modelToBeQuantizedPath)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    def representative_dataset():
        # Generator function for representative dataset used in quantization
        for i in range(500):
            yield [X_train[i].astype(np.float32)]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32  # or tf.uint8
    converter.inference_output_type = tf.float32  # or tf.uint8
    quantizedModelTFlite = converter.convert()
    with open(quantizedModelPath, 'wb') as f:
        f.write(quantizedModelTFlite)


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


def getTFlitePredictions8INT(path_to_tflite_model, X_data):
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
        x = x.astype(np.int8).reshape(1, 28, 28)
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

