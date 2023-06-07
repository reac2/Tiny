import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import optuna
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler



class GeneralNNRegressor:
    def __init__(self, X, y):
        """
        General Neural Network (NN) regressor.

        Args:
            X (numpy.ndarray): The input data.
            y (numpy.ndarray): The target data.
        """
        self.X = X
        self.y = y

    def get_best_trained_model(self, n_trials=50):
        """
        Get the best trained model by finding the best model parameters and training the model.

        Args:
            n_trials (int, optional): The number of trials for hyperparameter optimization. Defaults to 50.

        Returns:
            tf.keras.Model: The best trained model.
        """
        best_network_params = self.find_best_model_params(n_trials=n_trials)
        best_model = self.make_model(best_network_params)
        best_model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=best_network_params['learning_rate'])
        )
        best_model.fit(self.X, self.y, epochs=50, batch_size=64)
        print('Best model size:', self.count_trainable_parameters(best_model))
        return best_model

    def objective(self, trial, n_splits=5):
        """
        Objective function for hyperparameter optimization using Optuna.

        Args:
            trial (optuna.Trial): The Optuna trial object.
            n_splits (int, optional): The number of splits for cross-validation. Defaults to 5.

        Returns:
            float: The average loss score over the cross-validation folds.
            int: The number of trainable parameters in the model.
        """
        # Suggest number of layers and learning rate
        network_params = {
            'num_layers': trial.suggest_int('num_layers', 2, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1)
        }

        # Loop over suggested number of layers
        for i in range(network_params['num_layers']):
            # Suggest number of nodes in the layer
            network_params[f'node_{i}'] = trial.suggest_int(f'node_{i}', 4, 256)

        print(network_params)
        score = 0
        for _, (train_idx, test_idx) in enumerate(KFold(n_splits=n_splits, shuffle=True).split(self.X, self.y)):
            model = self.make_model(network_params)
            model.compile(
                loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=network_params['learning_rate'])
            )
            model.fit(self.X[train_idx], self.y[train_idx], validation_split=0.2, epochs=50, batch_size=64, verbose=0)
            score += model.evaluate(self.X[test_idx], self.y[test_idx], verbose=0)

        print(f'Score: {score / n_splits}, Model size: {self.count_trainable_parameters(model)}')
        return score / n_splits, self.count_trainable_parameters(model)

    def find_best_model_params(self, n_trials):
        """
        Find the best model parameters using Optuna's hyperparameter optimization.

        Args:
            n_trials (int): The number of trials for optimization.

        Returns:
            dict: The best model parameters.
        """
        study = optuna.create_study(directions=['minimize', 'minimize'])
        study.optimize(self.objective, n_trials)

        vals = np.array([[i.values[0], i.values[1]] for i in study.trials])
        vals[:, 0] = 1 - (vals[:, 0] - vals[:, 0].min()) / (vals[:, 0].max() - vals[:, 0].min())
        vals[:, 1] = 1 - ((vals[:, 1] - vals[:, 1].min()) / (vals[:, 1].max() - vals[:, 1].min()))
        comb_err = vals[:, 0] + 0.2 * vals[:, 1]

        return study.trials[np.argmax(comb_err)].params

    def make_model(self, network_params):
        """
        Create a neural network model based on the given parameters.

        Args:
            network_params (dict): The parameters for configuring the neural network model.

        Returns:
            tf.keras.Model: The neural network model.
        """
        model = tf.keras.Sequential([layers.Input(self.X.shape[1:])])
        # Is the data multidimensional?
        if len(self.X.shape[1:]) > 1:
            # Add a flatten layer
            model.add(layers.Flatten())

        # Loop over layers
        for i in range(network_params['num_layers']):
            # Add a dense layer with the specified number of nodes and activation
            model.add(layers.Dense(network_params[f'node_{i}'], activation='relu'))

        model.add(layers.Dense(1))
        return model

    @staticmethod
    def count_trainable_parameters(model):
        """
        Count the number of trainable parameters in the given model.

        Args:
            model (tf.keras.Model): The model to count the trainable parameters.

        Returns:
            int: The number of trainable parameters.
        """
        trainable_params = tf.reduce_sum([tf.reduce_prod(var.shape) for var in model.trainable_variables])
        return trainable_params.numpy()
    
class GeneralNNClassifier:
    def __init__(self, X, y, n_trials=50):
        """
        General Neural Network (NN) classifier.

        Args:
            X (numpy.ndarray): The input data.
            y (numpy.ndarray): The target data.
            n_trials (int, optional): The number of trials for hyperparameter optimization. Defaults to 50.
        """
        self.X = X
        self.num_classes = len(np.unique(y))
        self.y = tf.keras.utils.to_categorical(y)
        self.n_trials = n_trials 

    def get_best_trained_model(self):
        """
        Get the best trained model by finding the best model parameters and training the model.

        Returns:
            tf.keras.Model: The best trained model.
        """
        best_network_params = self.find_best_model_params(n_trials=self.n_trials)
        best_model = self.make_model(best_network_params)
        best_model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=best_network_params['learning_rate']),
            metrics=['accuracy']
        )
        best_model.fit(self.X, self.y, epochs=10, batch_size=64)
        print('Best model size:', self.count_trainable_parameters(best_model))
        return best_model


    def objective(self, trial, n_splits=5):
        """
        Objective function for hyperparameter optimization using Optuna.

        Args:
            trial (optuna.Trial): The Optuna trial object.
            n_splits (int, optional): The number of splits for cross-validation. Defaults to 5.

        Returns:
            float: The average accuracy score over the cross-validation folds.
            int: The number of trainable parameters in the model.
        """
        # Suggest number of layers and learning rate
        network_params = {
            'num_layers': trial.suggest_int('num_layers', 2, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1)
        }

        # Loop over suggested number of layers
        for i in range(network_params['num_layers']):
            # Suggest number of nodes in the layer
            network_params[f'node_{i}'] = trial.suggest_int(f'node_{i}', 4, 512)

        print(network_params)
        score = 0
        for _, (train_idx, test_idx) in enumerate(KFold(n_splits=n_splits, shuffle=True).split(self.X, self.y.argmax(1))):
            model = self.make_model(network_params)
            model.compile(
                loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=network_params['learning_rate']),
                metrics=['accuracy']
            )
            model.fit(self.X[train_idx], self.y[train_idx], validation_split=0.2, epochs=5, batch_size=64, verbose=0)
            score += model.evaluate(self.X[test_idx], self.y[test_idx], verbose=0)[1]

        print(f'Score: {score / n_splits}, Model size: {self.count_trainable_parameters(model)}')
        return score / n_splits, self.count_trainable_parameters(model)
    
    def find_best_model_params(self, n_trials):
        """
        Find the best model parameters using Optuna's hyperparameter optimization.

        Args:
            n_trials (int): The number of trials for optimization.

        Returns:
            dict: The best model parameters.
        """
        study = optuna.create_study(directions=['maximize', 'minimize'])
        study.optimize(self.objective, n_trials)

        vals = np.array([[i.values[0], i.values[1]] for i in study.trials])
        vals[:, 1] = 1 - ((vals[:, 1] - vals[:, 1].min()) / (vals[:, 1].max() - vals[:, 1].min()))
        comb_err = vals[:, 0] + 0.2 * vals[:, 1]
    
        return study.trials[np.argmax(comb_err)].params

    def make_model(self, network_params):
        """
        Create a neural network model based on the given parameters.

        Args:
            network_params (dict): The parameters for configuring the neural network model.

        Returns:
            tf.keras.Model: The neural network model.
        """
        model = tf.keras.Sequential([layers.Input(self.X.shape[1:])])
        
        # Is the data multidimensional?
        if len(self.X.shape[1:]) > 1:
            # Add a flatten layer
            model.add(layers.Flatten())
        
        # Loop over layers
        for i in range(network_params['num_layers']):
            # Add a dense layer with the specified number of nodes and activation
            model.add(layers.Dense(network_params[f'node_{i}'], activation='relu'))
        
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        return model 
    
    @staticmethod
    def count_trainable_parameters(model):
        """
        Count the number of trainable parameters in the given model.

        Args:
            model (tf.keras.Model): The model to count the trainable parameters.

        Returns:
            int: The number of trainable parameters.
        """
        trainable_params = tf.reduce_sum([tf.reduce_prod(var.shape) for var in model.trainable_variables])
        return trainable_params.numpy()

class CNNClassifier:
    def __init__(self, X, y, n_trials=50, BATCH_SIZE=1024, EPOCHS=10):
        """
        Convolutional Neural Network (CNN) classifier.

        Args:
            X (numpy.ndarray): The input data.
            y (numpy.ndarray): The target data.
            n_trials (int, optional): The number of trials for hyperparameter optimization. Defaults to 50.
            BATCH_SIZE (int, optional): The batch size for training the model. Defaults to 1024.
        """
        self.X = X
        self.num_classes = len(np.unique(y))
        self.y = tf.keras.utils.to_categorical(y)
        self.n_trials = n_trials
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS

    def get_best_trained_model(self):
        """
        Get the best trained model by finding the best model parameters and training the model.

        Returns:
            tf.keras.Model: The best trained model.
        """
        best_network_params = self.find_best_model_params(n_trials=self.n_trials)
        best_model = self.make_model(best_network_params)
        best_model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=best_network_params['learning_rate']),
            metrics=['accuracy']
        )
        best_model.fit(self.X, self.y, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE)
        print('Best model size:', self.count_trainable_parameters(best_model))
        return best_model

    def objective(self, trial, n_splits=5):
        """
        Objective function for hyperparameter optimization using Optuna.

        Args:
            trial (optuna.Trial): The Optuna trial object.
            n_splits (int, optional): The number of splits for cross-validation. Defaults to 5.

        Returns:
            float: The average accuracy score over the cross-validation folds.
            int: The number of trainable parameters in the model.
        """
        network_params = {
            'num_conv_layers': trial.suggest_int('num_conv_layers', 1, 3),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1)
        }

        for i in range(network_params['num_conv_layers']):
            network_params[f'filters_{i}'] = trial.suggest_int(f'filters_{i}', 8, 128)
            network_params[f'kernel_size_{i}'] = trial.suggest_int(f'kernel_size_{i}', 3, 7)
            network_params[f'pool_size_{i}'] = trial.suggest_int(f'pool_size_{i}', 2, 4)

        print(network_params)
        score = 0
        for _, (train_idx, test_idx) in enumerate(KFold(n_splits=n_splits, shuffle=True).split(self.X, self.y.argmax(1))):
            model = self.make_model(network_params)
            model.compile(
                loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=network_params['learning_rate']),
                metrics=['accuracy']
            )
            model.fit(self.X[train_idx], self.y[train_idx], validation_split=0.2, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE, verbose=0)
            score += model.evaluate(self.X[test_idx], self.y[test_idx], verbose=0)[1]

        print(f'Score: {score / n_splits}, Model size: {self.count_trainable_parameters(model)}')
        return score / n_splits, self.count_trainable_parameters(model)

    def find_best_model_params(self, n_trials):
        """
        Find the best model parameters using Optuna's hyperparameter optimization.

        Args:
            n_trials (int): The number of trials for optimization.

        Returns:
            dict: The best model parameters.
        """
        study = optuna.create_study(directions=['maximize', 'minimize'])
        study.optimize(self.objective, n_trials)

        vals = np.array([[i.values[0], i.values[1]] for i in study.trials])
        vals[:, 1] = 1 - ((vals[:, 1] - vals[:, 1].min()) / (vals[:, 1].max() - vals[:, 1].min()))
        comb_err = vals[:, 0] + 0.2 * vals[:, 1]

        return study.trials[np.argmax(comb_err)].params

    def make_model(self, network_params):
        """
        Create a CNN model based on the given parameters.

        Args:
            network_params (dict): The parameters for configuring the CNN model.

        Returns:
            tf.keras.Model: The CNN model.
        """
        model = tf.keras.Sequential()

        # Reshape the input data to have a channel dimension
        model.add(layers.Reshape((self.X.shape[1], self.X.shape[2], 1), input_shape=(self.X.shape[1], self.X.shape[2])))

        for i in range(network_params['num_conv_layers']):
            model.add(layers.Conv2D(
                filters=network_params[f'filters_{i}'],
                kernel_size=network_params[f'kernel_size_{i}'],
                activation='relu',
                padding='same'
            ))

            # Get the output shape after convolution
            conv_output_shape = model.layers[-1].output_shape[1:]

            # Adjust the pool size if it exceeds the input size
            pool_size = min(network_params[f'pool_size_{i}'], *conv_output_shape)

            model.add(layers.MaxPooling2D(pool_size=pool_size))

        model.add(layers.Flatten())
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        return model

    @staticmethod
    def count_trainable_parameters(model):
        """
        Count the number of trainable parameters in the given model.

        Args:
            model (tf.keras.Model): The model to count the trainable parameters.

        Returns:
            int: The number of trainable parameters.
        """
        trainable_params = tf.reduce_sum([tf.reduce_prod(var.shape) for var in model.trainable_variables])
        return trainable_params.numpy()




if __name__ == '__main__':
    with tf.device('CPU:0'):
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data()
        # X_train = X_train[:10000] / 255
        # X_test = X_test / 255
        # y_train = y_train[:10000]
        scale = StandardScaler()
        X_train = scale.fit_transform(X_train)
        X_test = scale.transform(X_test)

        gen_net = GeneralNNRegressor(X_train, y_train)
        model = gen_net.get_best_trained_model()
        print('score', model.evaluate(X_test, y_test))
        print(model.summary())
    # model.compile(optimizer='adam', loss='categorical_crossentropy')
    # print(model.summary())