"""Utilities for calculating TreeSHAP feature contributions per position."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

class MulticlassShap:
	"""Per-class TreeSHAP explainers built once from an ONNX model."""

	def __init__(self, onnx_path, background, feature_columns, class_names=None, max_background=5000):
		import shap
		self.feature_columns = list(feature_columns)
		bg = background.loc[:, self.feature_columns]
		if len(bg) > max_background:
			bg = bg.sample(max_background, random_state=0)
		dicts = onnx_to_tree_dicts(onnx_path, bg)
		self.class_names = list(class_names) if class_names is not None else list(range(len(dicts)))
		self.explainers = [shap.TreeExplainer(d) for d in dicts]

	def class_index(self, label):
		if label in self.class_names:
			return self.class_names.index(label)
		try:
			return int(label)
		except (TypeError, ValueError):
			raise ValueError(
				f"predicted_label {label!r} is not one of {self.class_names}"
			) from None

	def explain(self, row_df, label):
		k = self.class_index(label)
		x = _as_float32_array(row_df.loc[:, self.feature_columns])
		ex = self.explainers[k]
		vals = np.asarray(ex.shap_values(x, check_additivity=False))
		if vals.ndim == 3:
			vals = vals[:, :, 0]
		base = float(np.atleast_1d(ex.expected_value)[0])
		contributions = [
			{"feature": f, "value": float(v)}
			for f, v in zip(self.feature_columns, vals[0])
		]
		return contributions, base, self.class_names[k]


def _as_float32_array(data: Any) -> np.ndarray:
	"""Convert prediction-dataframe or array input to a numeric matrix."""
	if hasattr(data, "to_numpy"):
		data = data.to_numpy()
	return np.asarray(data, dtype=np.float32)


def _attrs(node: Any) -> dict:
	from onnx import helper
	return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def _node_covers(left, right, feat, thr, default, background) -> np.ndarray:
	"""Count background rows reaching each node (stand-in for training covers)."""
	cover = np.zeros(len(left))
	node = np.zeros(len(background), dtype=np.int64)
	np.add.at(cover, node, 1)
	while True:
		rows = np.where(left[node] != -1)[0]
		if rows.size == 0:
			break
		n = node[rows]
		x = background[rows, feat[n]]
		go_left = np.where(np.isnan(x), default[n] == left[n], x <= thr[n])
		node[rows] = np.where(go_left, left[n], right[n])
		np.add.at(cover, node[rows], 1)
	return np.maximum(cover, 1e-6)


def _decode(value: Any) -> str:
	return value.decode() if isinstance(value, bytes) else value


def _load_ensemble(onnx_path: str, background: Any):
	"""Parse tree structure + covers once; leaf values are added per class later."""
	import onnx

	bg = _as_float32_array(background)
	graph = onnx.load(onnx_path).graph
	node = next(
		(n for n in graph.node if n.op_type in ("TreeEnsembleClassifier", "TreeEnsembleRegressor")),
		None,
	)
	if node is None:
		raise ValueError("No TreeEnsembleClassifier/Regressor node found in the ONNX graph")
	a = _attrs(node)
	is_clf = node.op_type == "TreeEnsembleClassifier"

	post = _decode(a.get("post_transform", b"NONE"))
	if post not in ("NONE", "LOGISTIC", "SOFTMAX"):
		raise ValueError(f"Unsupported post_transform: {post}")

	tree_ids = np.asarray(a["nodes_treeids"])
	node_ids = np.asarray(a["nodes_nodeids"])
	missing_true = a.get("nodes_missing_value_tracks_true", [0] * len(tree_ids))

	structs = []
	for t in np.unique(tree_ids):
		idx = np.where(tree_ids == t)[0]
		idx = idx[np.argsort(node_ids[idx], kind="stable")]  # root must be local index 0
		local = {int(node_ids[i]): k for k, i in enumerate(idx)}
		n = len(idx)
		left, right, default = (np.full(n, -1, np.int32) for _ in range(3))
		feat = np.full(n, -2, np.int32)
		thr = np.zeros(n, np.float32)
		for k, i in enumerate(idx):
			mode = _decode(a["nodes_modes"][i])
			if mode == "LEAF":
				continue
			if mode not in ("BRANCH_LEQ", "BRANCH_LT"):
				raise ValueError(f"Unsupported node mode: {mode}")
			left[k] = local[int(a["nodes_truenodeids"][i])]
			right[k] = local[int(a["nodes_falsenodeids"][i])]
			feat[k] = a["nodes_featureids"][i]
			thr[k] = a["nodes_values"][i]
			if mode == "BRANCH_LT":  # shap uses <=, so nudge the threshold down
				thr[k] = np.nextafter(thr[k], np.float32(-np.inf))
			default[k] = left[k] if missing_true[i] else right[k]
		structs.append({
			"tree_id": int(t),
			"node_ids": node_ids[idx],
			"left": left, "right": right, "default": default,
			"feat": feat, "thr": thr,
			"cover": _node_covers(left, right, feat, thr, default, bg),
		})
	return a, is_clf, post, structs


def onnx_to_tree_dicts(
	onnx_path: str,
	background: Any,
	class_indices: Optional[Sequence[int]] = None,
) -> list[dict]:
	"""Convert an ONNX TreeEnsemble into one shap-readable dict per requested class.

	Binary models (and regressors) give one dict. Multiclass models give one dict
	per class, each explaining that class's pre-softmax margin (log-odds-like value).
	"""
	a, is_clf, post, structs = _load_ensemble(onnx_path, background)
	p = "class" if is_clf else "target"
	out_ids = np.asarray(a[f"{p}_ids"])
	n_classes = int(out_ids.max()) + 1

	# leaf_w[(tree, node)] = {output_id: weight}
	leaf_w: dict = {}
	for t, nd, o, w in zip(a[f"{p}_treeids"], a[f"{p}_nodeids"], out_ids, a[f"{p}_weights"]):
		d = leaf_w.setdefault((int(t), int(nd)), {})
		d[int(o)] = d.get(int(o), 0.0) + float(w)

	two_class_softmax = post == "SOFTMAX" and n_classes == 2
	is_logit = post == "LOGISTIC" or two_class_softmax

	if class_indices is None:
		class_indices = [n_classes - 1] if n_classes <= 2 else list(range(n_classes))
	base = np.atleast_1d(a.get("base_values", [0.0])).astype(np.float64)

	dicts = []
	for target in class_indices:
		target = int(target)
		if not 0 <= target < n_classes:
			raise ValueError(f"class_index {target} out of range for {n_classes} outputs")
		other = (1 - target) if two_class_softmax else None  # margin = target - other

		trees = []
		for s in structs:
			value = np.zeros((len(s["left"]), 1))
			for k, nid in enumerate(s["node_ids"]):
				if s["left"][k] != -1:
					continue
				w = leaf_w.get((s["tree_id"], int(nid)), {})
				v = w.get(target, 0.0)
				if other is not None:
					v -= w.get(other, 0.0)
				value[k, 0] = v
			trees.append({
				"children_left": s["left"], "children_right": s["right"],
				"children_default": s["default"],
				"feature": s["feat"], "threshold": s["thr"], "value": value,
				"node_sample_weight": s["cover"],
			})

		base_offset = float(base[target]) if len(base) > target else float(base[0])
		if two_class_softmax and len(base) == 2:
			base_offset = float(base[target] - base[1 - target])

		dicts.append({
			"trees": trees,
			"base_offset": base_offset,
			"objective": "binary" if is_logit else "regression",
			"tree_output": "log_odds" if is_logit else "raw_value",
			"input_dtype": np.float32,
			"internal_dtype": np.float64,
		})
	return dicts


def onnx_to_tree_dict(onnx_path: str, background: Any, class_index: Optional[int] = None) -> dict:
	"""Single-output convenience wrapper (binary / regression / one chosen class)."""
	dicts = onnx_to_tree_dicts(
		onnx_path, background, None if class_index is None else [class_index]
	)
	if len(dicts) != 1:
		raise ValueError("Multiclass model: pass class_index, or use calculate_shap_features_multiclass")
	return dicts[0]


def _resolve_tree_model(model: Any) -> Any:
	"""Return the native tree model from a wrapper (LightGBM/sklearn/XGBoost)."""
	if isinstance(model, dict):  # pre-built ONNX tree dict
		return model
	if isinstance(model, str):
		if model.endswith(".onnx"):
			raise TypeError("Convert ONNX paths with onnx_to_tree_dict(...) first")
		if model.endswith(".txt"):
			import lightgbm as lgb
			return lgb.Booster(model_file=model)
		import joblib
		return joblib.load(model)

	for attr in ("booster_", "native_model", "lgbm_model", "classifier", "estimator", "model"):
		inner = getattr(model, attr, None)
		if inner is not None and not callable(inner):
			return inner

	if hasattr(model, "feature_importances_") or hasattr(model, "predict"):
		module = type(model).__module__.split(".")[0]
		if module in {"lightgbm", "sklearn", "xgboost", "catboost"}:
			return model

	raise TypeError(
		f"{type(model).__name__} has no native tree model "
		f"(attributes: {sorted(vars(model))}). Pass a tree dict from onnx_to_tree_dict, "
		"the original LightGBM/sklearn model, or a path to it."
	)


def get_tree_shap_values(
	tree_model: Any,
	input_data: Any,
	*,
	output_index: int = 1,
	background_data: Optional[Any] = None,
	check_additivity: bool = False,
) -> tuple[np.ndarray, float]:
	"""Return (shap_values[n_positions, n_features], base_value) via TreeSHAP."""
	try:
		import shap
	except ImportError as exc:  # pragma: no cover - depends on deployment
		raise ImportError("The 'shap' package is required to calculate SHAP values") from exc

	inputs = _as_float32_array(input_data)
	if inputs.ndim == 1:
		inputs = inputs.reshape(1, -1)

	if background_data is None:
		# Path-dependent TreeSHAP: no background needed, fastest option.
		explainer = shap.TreeExplainer(tree_model)
	else:
		explainer = shap.TreeExplainer(
			tree_model,
			data=_as_float32_array(background_data),
			feature_perturbation="interventional",
		)

	values = explainer.shap_values(inputs, check_additivity=check_additivity)
	expected = explainer.expected_value

	# Older shap: list of per-class arrays. Newer shap: (n, features, classes).
	if isinstance(values, list):
		idx = min(output_index, len(values) - 1)
		values = values[idx]
		expected = np.atleast_1d(expected)[idx]
	else:
		values = np.asarray(values)
		if values.ndim == 3:
			idx = min(output_index, values.shape[2] - 1)
			values = values[:, :, idx]
			expected = np.atleast_1d(expected)[idx]
		else:
			expected = np.atleast_1d(expected)[0]

	return np.asarray(values, dtype=np.float64), float(expected)


def calculate_shap_features_per_position(
	model: Any,
	prediction_dataframe: Any,
	feature_columns: Sequence[str],
	position_column: Optional[str] = None,
	output_index: int = 1,
	background_data: Optional[Any] = None,
) -> list[dict[str, Any]]:
	"""Calculate named TreeSHAP contributions for every position (row)."""
	missing = [
		column for column in feature_columns
		if column not in prediction_dataframe.columns
	]
	if missing:
		raise ValueError(f"Prediction dataframe is missing feature columns: {missing}")
	if len(prediction_dataframe) == 0:
		return []

	values, base_value = get_tree_shap_values(
		_resolve_tree_model(model),
		prediction_dataframe.loc[:, list(feature_columns)],
		output_index=output_index,
		background_data=background_data,
	)

	if position_column is not None:
		positions = prediction_dataframe[position_column].tolist()
	else:
		positions = list(range(len(prediction_dataframe)))

	return [
		{
			"position": position,
			"base_value": base_value,
			"contributions": [
				{"feature": feature, "value": float(value)}
				for feature, value in zip(feature_columns, row_values)
			],
		}
		for position, row_values in zip(positions, values)
	]


def compute_multiclass_shap(
	onnx_path: str,
	prediction_dataframe: Any,
	feature_columns: Sequence[str],
	background_data: Optional[Any] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Return (shap[class,row,feature], bases[class], outputs[class,row]).

	``outputs`` are the pre-softmax margins, so softmax(outputs, axis=0) should
	reproduce the ONNX class probabilities.
	"""
	X = prediction_dataframe.loc[:, list(feature_columns)]
	bg = X if background_data is None else background_data
	vals, bases = [], []
	for tree_dict in onnx_to_tree_dicts(onnx_path, bg):
		v, b = get_tree_shap_values(tree_dict, X, output_index=0)
		vals.append(v)
		bases.append(b)
	vals = np.stack(vals)
	bases = np.asarray(bases)
	outputs = bases[:, None] + vals.sum(axis=2)
	return vals, bases, outputs


def calculate_shap_features_multiclass(
	onnx_path: str,
	prediction_dataframe: Any,
	feature_columns: Sequence[str],
	class_names: Optional[Sequence[str]] = None,
	position_column: Optional[str] = None,
	background_data: Optional[Any] = None,
	class_index: Optional[int] = None,
) -> list[dict[str, Any]]:
	"""Per-position margin (logit) SHAP.

	By default each row reports its predicted class; pass ``class_index`` to
	report one fixed class for every row instead.
	"""
	missing = [c for c in feature_columns if c not in prediction_dataframe.columns]
	if missing:
		raise ValueError(f"Prediction dataframe is missing feature columns: {missing}")
	if len(prediction_dataframe) == 0:
		return []

	vals, bases, outputs = compute_multiclass_shap(
		onnx_path, prediction_dataframe, feature_columns, background_data
	)
	n_classes = vals.shape[0]
	names = list(class_names) if class_names is not None else list(range(n_classes))
	chosen = (
		np.full(vals.shape[1], int(class_index))
		if class_index is not None else outputs.argmax(axis=0)
	)
	positions = (
		prediction_dataframe[position_column].tolist()
		if position_column is not None else range(vals.shape[1])
	)
	return [
		{
			"position": pos,
			"class": names[k],
			"base_value": float(bases[k]),
			"output": float(outputs[k, i]),
			"contributions": [
				{"feature": f, "value": float(v)}
				for f, v in zip(feature_columns, vals[k, i])
			],
		}
		for i, (pos, k) in enumerate(zip(positions, chosen))
	]


def calculate_shap_features(
	model: Any,
	prediction_dataframe: Any,
	feature_columns: Sequence[str],
	row_index: int = 0,
	**kwargs: Any,
) -> list[dict[str, float | str]]:
	"""Backward-compatible single-row wrapper."""
	per_position = calculate_shap_features_per_position(
		model, prediction_dataframe, feature_columns, **kwargs
	)
	return per_position[row_index]["contributions"] if per_position else []
