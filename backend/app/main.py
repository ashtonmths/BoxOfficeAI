from __future__ import annotations

import ast
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("boxoffice-api")


class PredictRequest(BaseModel):
	genre: str = Field(min_length=1)
	year: int = Field(ge=1888, le=2100)
	runtime: float = Field(gt=0)
	budget: float = Field(ge=0)
	director: str = Field(min_length=1)
	actor: str = Field(min_length=1)
	votes: int = Field(ge=0)


class PredictResponse(BaseModel):
	predicted_revenue: float
	hit_probability: float
	prediction: str


class OptionsResponse(BaseModel):
	genres: List[str]
	actors: List[str]
	directors: List[str]


class PredictionService:
	def __init__(self) -> None:
		self.app_dir = Path(__file__).resolve().parent
		self.backend_dir = self.app_dir.parent
		self.project_dir = self.backend_dir.parent

		self.model_dir = self._resolve_existing_dir(
			[
				self.backend_dir / "model",
				self.project_dir / "model",
			]
		)

		self.data_dir = self._resolve_existing_dir(
			[
				self.project_dir / "data",
				self.backend_dir / "data",
			]
		)

		self.regressor = self._load_required_model(
			[
				self.model_dir / "regressor.pkl",
			]
		)
		self.classifier = self._load_required_model(
			[
				self.model_dir / "classifier.pkl",
			]
		)
		self.genre_encoder = self._load_required_model(
			[
				self.model_dir / "genre_encoder.pkl",
			]
		)

		self.scaler_cls = self._load_optional_model(
			[
				self.model_dir / "scaler_cls.pkl",
				self.model_dir / "scaler_cls" / "reg.pkl",
			]
		)
		self.scaler_reg = self._load_optional_model(
			[
				self.model_dir / "scaler_reg.pkl",
				self.model_dir / "scaler_reg" / "reg.pkl",
			]
		)

		# If only one scaler is available, use it for both models.
		if self.scaler_cls is None and self.scaler_reg is not None:
			self.scaler_cls = self.scaler_reg
		if self.scaler_reg is None and self.scaler_cls is not None:
			self.scaler_reg = self.scaler_cls

		self.credits_df = self._load_required_csv([self.data_dir / "credits.csv"])
		self.movies_df = self._load_required_csv([self.data_dir / "movies.csv"])

		self.min_year = self._compute_min_year(self.movies_df)
		self.genre_list = sorted([str(x) for x in getattr(self.genre_encoder, "classes_", [])])
		self.genre_lookup = {g.casefold(): g for g in self.genre_list}

		(
			self.actor_options,
			self.director_options,
			self.actor_power_map,
			self.director_power_map,
			self.actor_power_median,
			self.director_power_median,
		) = self._build_people_data(self.credits_df, self.movies_df)

		logger.info(
			"Prediction service initialized. genres=%d actors=%d directors=%d",
			len(self.genre_list),
			len(self.actor_options),
			len(self.director_options),
		)

	@staticmethod
	def _resolve_existing_dir(candidates: List[Path]) -> Path:
		for candidate in candidates:
			if candidate.exists() and candidate.is_dir():
				return candidate
		choices = " | ".join(str(c) for c in candidates)
		raise FileNotFoundError(f"No valid directory found. Tried: {choices}")

	@staticmethod
	def _load_required_csv(candidates: List[Path]) -> pd.DataFrame:
		for path in candidates:
			if path.exists():
				return pd.read_csv(path, low_memory=False)
		choices = " | ".join(str(c) for c in candidates)
		raise FileNotFoundError(f"CSV file not found. Tried: {choices}")

	@staticmethod
	def _load_required_model(candidates: List[Path]) -> Any:
		for path in candidates:
			if path.exists():
				return joblib.load(path)
		choices = " | ".join(str(c) for c in candidates)
		raise FileNotFoundError(f"Model file not found. Tried: {choices}")

	@staticmethod
	def _load_optional_model(candidates: List[Path]) -> Optional[Any]:
		for path in candidates:
			if path.exists():
				return joblib.load(path)
		return None

	@staticmethod
	def _normalize_name(name: str) -> str:
		return " ".join(str(name).split()).casefold()

	@staticmethod
	def _safe_parse_people_blob(blob: Any) -> List[Dict[str, Any]]:
		if pd.isna(blob):
			return []
		text = str(blob).strip()
		if not text:
			return []
		try:
			parsed = ast.literal_eval(text)
		except (ValueError, SyntaxError):
			return []
		if not isinstance(parsed, list):
			return []
		return [item for item in parsed if isinstance(item, dict)]

	@classmethod
	def _extract_cast_names(cls, cast_blob: Any) -> List[str]:
		cast_list = cls._safe_parse_people_blob(cast_blob)
		names = []
		for member in cast_list:
			name = str(member.get("name", "")).strip()
			if name:
				names.append(name)
		return names

	@classmethod
	def _extract_director_names(cls, crew_blob: Any) -> List[str]:
		crew_list = cls._safe_parse_people_blob(crew_blob)
		names = []
		for member in crew_list:
			name = str(member.get("name", "")).strip()
			if not name:
				continue
			job = str(member.get("job", "")).strip().casefold()
			if job == "director":
				names.append(name)
		return names

	@staticmethod
	def _compute_min_year(movies_df: pd.DataFrame) -> int:
		if "release_date" in movies_df.columns:
			release_date = pd.to_datetime(movies_df["release_date"], errors="coerce")
			valid_years = release_date.dt.year.dropna()
			if not valid_years.empty:
				return int(valid_years.min())

		if "year" in movies_df.columns:
			years = pd.to_numeric(movies_df["year"], errors="coerce").dropna()
			if not years.empty:
				return int(years.min())

		return 1900

	def _build_people_data(
		self, credits_df: pd.DataFrame, movies_df: pd.DataFrame
	) -> Tuple[List[str], List[str], Dict[str, float], Dict[str, float], float, float]:
		actor_counter: Counter[str] = Counter()
		director_counter: Counter[str] = Counter()
		actor_sums: Dict[str, float] = defaultdict(float)
		actor_counts: Dict[str, int] = defaultdict(int)
		director_sums: Dict[str, float] = defaultdict(float)
		director_counts: Dict[str, int] = defaultdict(int)

		revenue_by_id: Dict[int, float] = {}
		if {"id", "revenue"}.issubset(movies_df.columns):
			movies_small = movies_df[["id", "revenue"]].copy()
			movies_small["id"] = pd.to_numeric(movies_small["id"], errors="coerce")
			movies_small["revenue"] = pd.to_numeric(movies_small["revenue"], errors="coerce")
			movies_small = movies_small.dropna(subset=["id", "revenue"])
			movies_small = movies_small[movies_small["revenue"] > 0]
			revenue_by_id = {
				int(row.id): float(row.revenue)
				for row in movies_small.itertuples(index=False)
			}

		for row in credits_df.itertuples(index=False):
			movie_id_raw = getattr(row, "id", None)
			cast_blob = getattr(row, "cast", None)
			crew_blob = getattr(row, "crew", None)

			cast_names = self._extract_cast_names(cast_blob)
			director_names = self._extract_director_names(crew_blob)

			actor_counter.update(cast_names)
			director_counter.update(director_names)

			try:
				movie_id = int(movie_id_raw)
			except (TypeError, ValueError):
				continue

			revenue = revenue_by_id.get(movie_id)
			if revenue is None:
				continue

			for name in set(cast_names):
				key = self._normalize_name(name)
				actor_sums[key] += revenue
				actor_counts[key] += 1

			for name in set(director_names):
				key = self._normalize_name(name)
				director_sums[key] += revenue
				director_counts[key] += 1

		top_actors = [name for name, _ in actor_counter.most_common(1000)]
		top_directors = [name for name, _ in director_counter.most_common(1000)]

		actor_power_map = {
			name: actor_sums[name] / actor_counts[name]
			for name in actor_sums
			if actor_counts[name] > 0
		}
		director_power_map = {
			name: director_sums[name] / director_counts[name]
			for name in director_sums
			if director_counts[name] > 0
		}

		actor_median = (
			float(np.median(list(actor_power_map.values())))
			if actor_power_map
			else float(np.median(list(revenue_by_id.values())))
			if revenue_by_id
			else 1.0
		)

		director_median = (
			float(np.median(list(director_power_map.values())))
			if director_power_map
			else float(np.median(list(revenue_by_id.values())))
			if revenue_by_id
			else 1.0
		)

		return (
			top_actors,
			top_directors,
			actor_power_map,
			director_power_map,
			actor_median,
			director_median,
		)

	def _encode_genre(self, genre: str) -> int:
		key = genre.strip().casefold()
		canonical = self.genre_lookup.get(key)
		if canonical is None:
			sample = ", ".join(self.genre_list[:10])
			raise ValueError(
				f"Unknown genre '{genre}'. Use GET /options for valid values. Sample: {sample}"
			)
		return int(self.genre_encoder.transform([canonical])[0])

	def _build_feature_dict(self, payload: PredictRequest) -> Dict[str, float]:
		actor_key = self._normalize_name(payload.actor)
		director_key = self._normalize_name(payload.director)
		genre_encoded = float(self._encode_genre(payload.genre))

		actor_power = self.actor_power_map.get(actor_key, self.actor_power_median)
		director_power = self.director_power_map.get(
			director_key, self.director_power_median
		)

		year_scaled = float(payload.year - self.min_year)
		runtime = float(payload.runtime)
		budget = float(payload.budget)
		votes = float(payload.votes)

		return {
			# Keep both training column names and explicit transformed aliases.
			"genre": genre_encoded,
			"genre_encoded": genre_encoded,
			"year": year_scaled,
			"year_scaled": year_scaled,
			"runtime": runtime,
			"log_runtime": float(np.log1p(runtime)),
			"log_budget": float(np.log1p(budget)),
			"budget_runtime": float(budget / (runtime + 1.0)),
			"budget_year": float(budget * year_scaled),
			"log_votes": float(np.log1p(votes)),
			"log_actor_power": float(np.log1p(actor_power)),
			"log_director_power": float(np.log1p(director_power)),
		}

	@staticmethod
	def _resolve_feature_order(
		feature_dict: Dict[str, float], model: Any, scaler: Optional[Any]
	) -> List[str]:
		canonical_order = [
			"genre",
			"year",
			"runtime",
			"log_runtime",
			"log_budget",
			"budget_runtime",
			"budget_year",
			"log_director_power",
			"log_actor_power",
			"log_votes",
			"genre_encoded",
			"year_scaled",
		]

		fallback_order = [
			"genre_encoded",
			"year_scaled",
			"log_votes",
			"log_actor_power",
			"log_director_power",
		]

		if scaler is not None and hasattr(scaler, "feature_names_in_"):
			return [str(col) for col in scaler.feature_names_in_]

		if hasattr(model, "feature_names_in_"):
			return [str(col) for col in model.feature_names_in_]

		if scaler is not None and hasattr(scaler, "n_features_in_"):
			n_features = int(scaler.n_features_in_)
			if 0 < n_features <= len(canonical_order):
				return canonical_order[:n_features]

		if hasattr(model, "n_features_in_"):
			n_features = int(model.n_features_in_)
			if 0 < n_features <= len(canonical_order):
				return canonical_order[:n_features]

		resolved = [col for col in canonical_order if col in feature_dict]
		if resolved:
			return resolved
		return [col for col in fallback_order if col in feature_dict]

	def _build_model_matrix(
		self, feature_dict: Dict[str, float], model: Any, scaler: Optional[Any]
	) -> np.ndarray:
		feature_order = self._resolve_feature_order(feature_dict, model, scaler)
		missing_cols = [col for col in feature_order if col not in feature_dict]
		if missing_cols:
			raise ValueError(f"Missing feature columns: {missing_cols}")

		frame = pd.DataFrame([{col: feature_dict[col] for col in feature_order}])

		if scaler is None:
			return frame.to_numpy(dtype=float)

		scaled = scaler.transform(frame)
		return np.asarray(scaled, dtype=float)

	def predict(self, payload: PredictRequest) -> PredictResponse:
		feature_dict = self._build_feature_dict(payload)

		regression_X = self._build_model_matrix(feature_dict, self.regressor, self.scaler_reg)
		classification_X = self._build_model_matrix(
			feature_dict, self.classifier, self.scaler_cls
		)

		regression_log = float(np.ravel(self.regressor.predict(regression_X))[0])
		predicted_revenue = float(max(0.0, np.expm1(regression_log)))

		probabilities = np.ravel(self.classifier.predict_proba(classification_X)).astype(float)
		classes = list(getattr(self.classifier, "classes_", []))

		hit_index = len(probabilities) - 1
		if classes:
			if 1 in classes:
				hit_index = classes.index(1)
			else:
				for idx, cls_name in enumerate(classes):
					if str(cls_name).strip().casefold() == "hit":
						hit_index = idx
						break

		hit_probability = float(np.clip(probabilities[hit_index], 0.0, 1.0))
		label = "Hit" if hit_probability >= 0.5 else "Flop"

		return PredictResponse(
			predicted_revenue=predicted_revenue,
			hit_probability=hit_probability,
			prediction=label,
		)

	def get_options(self) -> OptionsResponse:
		return OptionsResponse(
			genres=self.genre_list,
			actors=self.actor_options,
			directors=self.director_options,
		)


app = FastAPI(
	title="BoxOfficeAI Prediction API",
	version="1.0.0",
	description="Production-ready API for movie revenue and hit prediction.",
)

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


prediction_service: Optional[PredictionService] = None


@app.on_event("startup")
def startup_event() -> None:
	global prediction_service
	prediction_service = PredictionService()


def get_service() -> PredictionService:
	if prediction_service is None:
		raise HTTPException(status_code=503, detail="Model service is not initialized")
	return prediction_service


@app.get("/health")
def health() -> Dict[str, Any]:
	service_ready = prediction_service is not None
	return {
		"status": "ok" if service_ready else "initializing",
		"service_ready": service_ready,
	}


@app.get("/options", response_model=OptionsResponse)
def options() -> OptionsResponse:
	return get_service().get_options()


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
	try:
		return get_service().predict(payload)
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except HTTPException:
		raise
	except Exception as exc:
		logger.exception("Prediction failed")
		raise HTTPException(status_code=500, detail="Prediction failed") from exc