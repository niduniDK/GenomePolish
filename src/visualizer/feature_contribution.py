"""Utilities for calculating feature contributions for ONNX predictions."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np


def calculate_shap_features(
	model: Any,
	prediction_dataframe: Any,
	feature_columns: Sequence[str],
	row_index: int = 0,
	max_evals: int = 2048,
) -> list[dict[str, float | str]]:
	"""Calculate named SHAP contributions for one prediction row."""
	missing = [
		column for column in feature_columns
		if column not in prediction_dataframe.columns
	]
	if missing:
		raise ValueError(f"Prediction dataframe is missing feature columns: {missing}")
	if len(prediction_dataframe) == 0:
		return []

	values = get_shap_values(
		getattr(model, "session", model),
		prediction_dataframe.loc[:, list(feature_columns)],
		max_evals=max_evals,
	)
	row_values = np.asarray(values, dtype=np.float64)[row_index]
	return [
		{"feature": feature, "value": float(value)}
		for feature, value in zip(feature_columns, row_values)
	]


def _as_float32_array(data: Union[np.ndarray, list, Any]) -> np.ndarray:
	"""Convert prediction-dataframe or array input to a numeric matrix."""
	# Avoid importing pandas here: accepting the dataframe's public ``to_numpy``
	# interface keeps this module usable in lightweight browser-facing services.
	if hasattr(data, "to_numpy"):
		data = data.to_numpy()
	return np.asarray(data, dtype=np.float32)


def get_shap_values(
	onnx_model: Any,
	input_data: Union[np.ndarray, list],
	background_data: Optional[Union[np.ndarray, list]] = None,
	*,
	output_index: int = 0,
	max_evals: int = 2048,
) -> list:
	"""Return SHAP values for predictions made by an ONNX model.

	``onnx_model`` may be an ONNX Runtime ``InferenceSession`` or a path to
	an ONNX model.  The returned value is a plain list so it can be passed
	directly to a browser-rendering layer (for example, as JSON).
	"""
	try:
		import shap
	except ImportError as exc:  # pragma: no cover - depends on deployment
		raise ImportError("The 'shap' package is required to calculate SHAP values") from exc

	try:
		import onnxruntime as ort
	except ImportError as exc:  # pragma: no cover - depends on deployment
		raise ImportError("The 'onnxruntime' package is required for ONNX models") from exc

	session = onnx_model
	if isinstance(onnx_model, (str, bytes)):
		session = ort.InferenceSession(onnx_model)

	# ``input_data`` may be the prediction dataframe created by polishml.py.
	# Convert it once to a dense matrix so the result can be serialized and sent
	# to the WebAssembly rendering layer without pandas objects in the payload.
	inputs = _as_float32_array(input_data)
	if inputs.ndim == 1:
		inputs = inputs.reshape(1, -1)
	background = inputs if background_data is None else _as_float32_array(background_data)
	if background.ndim == 1:
		background = background.reshape(1, -1)

	input_name = session.get_inputs()[0].name

	def predict(values: np.ndarray) -> np.ndarray:
		result = session.run(None, {input_name: np.asarray(values, dtype=np.float32)})[output_index]
		result = np.asarray(result)
		if result.ndim == 1:
			return result
		return result[:, output_index] if result.shape[1] > 1 else result[:, 0]

	explainer = shap.KernelExplainer(predict, background)
	values = explainer.shap_values(inputs, nsamples=max_evals)
	if isinstance(values, list):
		values = values[output_index]
	return np.asarray(values).tolist()
