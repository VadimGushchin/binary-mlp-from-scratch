# ruff: noqa: F401
# pyright: ignore[reportMissingImports]
# type: ignore

import numpy as np
from sklearn.metrics import roc_auc_score

RANDOM_SEED = 42


class MLP:
    """
    Двухслойный перцептрон для бинарной классификации.
    Поддерживает различные функции активации, оптимизаторы (SGD, Adam)
    и L2-регуляризацию.
    """

    def __init__(
        self,
        n_hidden=100,
        activation_func="sigmoid",
        learning_rate=0.01,
        optimizer="sgd",
        l2_lambda=0.0,
    ):
        """
        Инициализация модели.

        Parameters
        ----------
        n_hidden : int, default=100
            Количество нейронов скрытого слоя.
        activation_func : str, default='sigmoid'
            Функция активации скрытого слоя:
            'sigmoid', 'relu', 'cos', 'tanh', 'elu'.
        learning_rate : float, default=0.01
            Скорость обучения.
        optimizer : str, default='sgd'
            Оптимизатор: 'sgd' или 'adam'.
        l2_lambda : float, default=0.0
            Коэффициент L2-регуляризации (weight decay).
        """
        self.n_hidden = n_hidden
        self.activation_func = activation_func
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.l2_lambda = l2_lambda

        # Параметры модели
        self.weights_input_hidden = None
        self.bias_hidden = None
        self.weights_hidden_output = None
        self.bias_output = None

        # Для Adam
        self.first_moment = {}
        self.second_moment = {}
        self.beta1, self.beta2, self.epsilon = 0.9, 0.999, 1e-8
        self.alpha = 1.0  # для ELU

    # ----------------------------------------------------------------------
    # Приватные методы
    # ----------------------------------------------------------------------
    def _get_scale(self, fan_in, fan_out, activation):
        """Возвращает масштаб инициализации весов в зависимости от активации."""
        if activation in ("sigmoid", "tanh"):
            return np.sqrt(2.0 / (fan_in + fan_out))
        if activation in ("relu", "cos", "elu", "softmax"):
            return np.sqrt(2.0 / fan_in)
        return 0.01

    def _initialize_weights(self, input_dim):
        """Инициализация весов и смещений."""
        np.random.seed(RANDOM_SEED)
        scale_hidden = self._get_scale(input_dim, self.n_hidden, self.activation_func)
        scale_output = self._get_scale(self.n_hidden, 1, self.activation_func)

        self.weights_input_hidden = (
            np.random.randn(input_dim, self.n_hidden) * scale_hidden
        )
        self.bias_hidden = np.zeros((1, self.n_hidden))
        self.weights_hidden_output = np.random.randn(self.n_hidden, 1) * scale_output
        self.bias_output = np.zeros((1, 1))

    def _activation(self, z):
        """Применяет функцию активации."""
        activations = {
            "sigmoid": lambda z: 1 / (1 + np.exp(-np.clip(z, -500, 500))),
            "relu": lambda z: np.maximum(0, z),
            "cos": np.cos,
            "tanh": np.tanh,
            "elu": lambda z: np.where(z > 0, z, self.alpha * (np.exp(z) - 1)),
        }
        if self.activation_func not in activations:
            raise ValueError(f"Unknown activation: {self.activation_func}")
        return activations[self.activation_func](z)

    def _activation_derivative(self, pre_act, act_out):
        """Производная функции активации по пред-активации."""
        derivatives = {
            "sigmoid": lambda p, a: a * (1 - a),
            "relu": lambda p, a: (p > 0).astype(float),
            "cos": lambda p, a: -np.sin(p),
            "tanh": lambda p, a: 1 - a**2,
            "elu": lambda p, a: np.where(p > 0, 1.0, a + self.alpha),
        }
        if self.activation_func not in derivatives:
            raise ValueError(f"Unknown activation: {self.activation_func}")
        return derivatives[self.activation_func](pre_act, act_out)

    def _forward(self, X):
        """Прямой проход: возвращает (hidden_linear, hidden_activation,
        output_linear, output_activation)."""
        hidden_linear = X @ self.weights_input_hidden + self.bias_hidden
        hidden_activation = self._activation(hidden_linear)

        output_linear = (
            hidden_activation @ self.weights_hidden_output + self.bias_output
        )
        output_activation = 1 / (1 + np.exp(-np.clip(output_linear, -500, 500)))

        return hidden_linear, hidden_activation, output_linear, output_activation

    def _binary_cross_entropy(self, y_true, y_pred):
        """Бинарная кросс-энтропия с L2-регуляризацией."""
        eps = 1e-15
        bce = -np.mean(
            y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)
        )
        l2_penalty = (
            0.5
            * self.l2_lambda
            * (
                np.sum(self.weights_input_hidden**2)
                + np.sum(self.weights_hidden_output**2)
            )
        )
        return bce + l2_penalty

    def _backward(
        self, X, y, hidden_linear, hidden_activation, output_linear, output_activation
    ):
        """Обратный проход: вычисляет градиенты параметров."""
        batch_size = X.shape[0]

        # Выходной слой
        output_error = output_activation - y.reshape(-1, 1)
        grad_weights_hidden_output = (hidden_activation.T @ output_error) / batch_size
        grad_bias_output = np.mean(output_error, axis=0, keepdims=True)

        # Скрытый слой
        hidden_error = (
            output_error @ self.weights_hidden_output.T
        ) * self._activation_derivative(hidden_linear, hidden_activation)
        grad_weights_input_hidden = (X.T @ hidden_error) / batch_size
        grad_bias_hidden = np.mean(hidden_error, axis=0, keepdims=True)

        # L2-градиенты
        grad_weights_input_hidden += self.l2_lambda * self.weights_input_hidden
        grad_weights_hidden_output += self.l2_lambda * self.weights_hidden_output

        return {
            "weights_input_hidden": grad_weights_input_hidden,
            "bias_hidden": grad_bias_hidden,
            "weights_hidden_output": grad_weights_hidden_output,
            "bias_output": grad_bias_output,
        }

    def _update_parameters(self, gradients, step=None):
        """Обновляет параметры с помощью SGD или Adam."""
        if self.optimizer == "sgd":
            for param_name, grad in gradients.items():
                setattr(
                    self,
                    param_name,
                    getattr(self, param_name) - self.learning_rate * grad,
                )
            return None

        if self.optimizer == "adam":
            if not self.first_moment:
                for k in gradients:
                    self.first_moment[k] = np.zeros_like(gradients[k])
                    self.second_moment[k] = np.zeros_like(gradients[k])

            step = step if step is not None else 1
            for k in gradients:
                self.first_moment[k] = (
                    self.beta1 * self.first_moment[k] + (1 - self.beta1) * gradients[k]
                )
                self.second_moment[k] = self.beta2 * self.second_moment[k] + (
                    1 - self.beta2
                ) * (gradients[k] ** 2)

                m_hat = self.first_moment[k] / (1 - self.beta1**step)
                v_hat = self.second_moment[k] / (1 - self.beta2**step)
                update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

                setattr(self, k, getattr(self, k) - update)

            return step + 1

        raise ValueError(f"Unknown optimizer: {self.optimizer}")

    def _batch_iterator(self, X, y, batch_size):
        """Генератор батчей с перемешиванием."""
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_indices = indices[start:end]
            yield X[batch_indices], y[batch_indices].reshape(-1, 1)

    def _train_epoch(self, X, y, batch_size, step=None):
        """Одна эпоха обучения: проход по всем батчам."""
        total_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in self._batch_iterator(X, y, batch_size):
            hidden_linear, hidden_activation, output_linear, output_pred = (
                self._forward(X_batch)
            )
            loss = self._binary_cross_entropy(y_batch, output_pred)
            total_loss += loss
            n_batches += 1

            gradients = self._backward(
                X_batch,
                y_batch,
                hidden_linear,
                hidden_activation,
                output_linear,
                output_pred,
            )
            step = self._update_parameters(gradients, step)

        return total_loss / n_batches, step

    def _compute_gini(self, X_val, y_val):
        """Вычисляет Gini = 2*AUC-1."""
        proba = self.predict_proba(X_val)[:, 1]
        return 2 * roc_auc_score(y_val, proba) - 1

    # ----------------------------------------------------------------------
    # Публичные методы
    # ----------------------------------------------------------------------
    def predict_proba(self, X):
        """
        Возвращает вероятности принадлежности к классам 0 и 1.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Входные данные.

        Returns
        -------
        np.ndarray, shape (n_samples, 2)
            Массив вероятностей: первый столбец – P(класс 0),
            второй – P(класс 1).
        """
        X = np.asarray(X)
        _, _, _, output_pred = self._forward(X)
        prob_class_1 = output_pred.ravel()
        return np.column_stack((1 - prob_class_1, prob_class_1))

    def predict(self, X, threshold=0.5):
        """
        Возвращает бинарные предсказания классов.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Входные данные.
        threshold : float, default=0.5
            Порог отсечения для класса 1.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Массив меток (0 или 1).
        """
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        epochs=100,
        batch_size=32,
        verbose=True,
        early_stopping_patience=30,
        early_stopping_min_delta=1e-4,
        restore_best_weights=True,
    ):
        """
        Обучение модели.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Обучающие признаки.
        y : array-like, shape (n_samples,)
            Целевые метки (0 или 1).
        X_val : array-like, optional, shape (n_samples, n_features)
            Валидационные признаки.
        y_val : array-like, optional, shape (n_samples,)
            Валидационные метки.
        epochs : int, default=100
            Максимальное число эпох.
        batch_size : int, default=32
            Размер батча.
        verbose : bool, default=True
            Выводить прогресс обучения.
        early_stopping_patience : int, default=30
            Число эпох без улучшения Gini до остановки.
        early_stopping_min_delta : float, default=1e-4
            Минимальное улучшение Gini для сброса счётчика.
        restore_best_weights : bool, default=True
            Восстанавливать ли веса с наилучшим валидационным Gini.

        Returns
        -------
        self : MLP
            Обученная модель.
        """
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        if X_val is not None:
            X_val = np.asarray(X_val)
        if y_val is not None:
            y_val = np.asarray(y_val).reshape(-1, 1)

        self._initialize_weights(X.shape[1])

        step = 1 if self.optimizer == "adam" else None
        self.first_moment = {}
        self.second_moment = {}

        best_gini = -np.inf
        patience_counter = 0
        best_weights = None

        for epoch in range(epochs):
            avg_loss, step = self._train_epoch(X, y, batch_size, step)

            if X_val is not None:
                current_gini = self._compute_gini(X_val, y_val)

                if verbose:
                    print(
                        f"Epoch {epoch + 1:3d} | Loss: {avg_loss:.4f} | "
                        f"Val Gini: {current_gini:.4f}"
                    )

                if current_gini > best_gini + early_stopping_min_delta:
                    best_gini = current_gini
                    patience_counter = 0
                    if restore_best_weights:
                        best_weights = {
                            "weights_input_hidden": self.weights_input_hidden.copy(),
                            "bias_hidden": self.bias_hidden.copy(),
                            "weights_hidden_output": self.weights_hidden_output.copy(),
                            "bias_output": self.bias_output.copy(),
                        }
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                if verbose:
                    print(f"Epoch {epoch + 1:3d} | Loss: {avg_loss:.4f}")

        if restore_best_weights and best_weights is not None:
            for key, value in best_weights.items():
                setattr(self, key, value)

        return self
