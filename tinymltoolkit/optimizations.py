import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np

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


def setQuantizedModel(modelToBeQuantizedPath,quantizedModelPath,X_train):
    converter = tf.lite.TFLiteConverter.from_saved_model(modelToBeQuantizedPath)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset(X_train=X_train)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    quantizedModelTFlite = converter.convert()
    with open(quantizedModelPath, 'wb') as f:
        f.write(quantizedModelTFlite)


def representative_dataset(X_train):
    # Generator function for representative dataset used in quantization
    for i in range(300):
        yield [X_train[i].astype(np.float32)]