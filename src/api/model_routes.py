"""
模型管理 API 路由

提供模型上傳、下載、重載等功能。
用於本地訓練環境將模型同步到 Railway 線上環境。
"""

import base64
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel

from src.config import settings
from src.logger import logger
from src.api.dependencies import verify_api_key

router = APIRouter(prefix="/api/v1/models", tags=["models"])

# 模型存儲目錄
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 已載入的模型快取
_loaded_models: Dict[str, Any] = {}

# iOS 本地訓練的模型快取
_ios_trained_models: Dict[str, Dict[str, Any]] = {}

# ML 模型路由（用於 iOS 本地訓練模型上傳）
ml_router = APIRouter(prefix="/api/v1/ml/models", tags=["ml-models"])


# ==================== 請求/響應模型 ====================

class ModelUploadRequest(BaseModel):
    """模型上傳請求"""
    filename: str
    data: str  # Base64 編碼的模型數據
    metadata: Optional[Dict[str, Any]] = None
    force: bool = False


class MLModelUploadRequest(BaseModel):
    """iOS 本地訓練模型上傳請求

    從 iOS App 本地訓練後推送的模型數據
    """
    symbol: str
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    weights: List[float]
    feature_importance: Dict[str, float]
    trained_at: str
    data_points: int


class MLModelUploadResponse(BaseModel):
    """ML 模型上傳響應"""
    success: bool
    message: str
    model_id: str
    registered_at: str


class ModelUploadResponse(BaseModel):
    """模型上傳響應"""
    success: bool
    message: str
    filename: str
    file_hash: str
    uploaded_at: str


class ModelMetadataRequest(BaseModel):
    """模型元數據請求"""
    filename: str
    metadata: Dict[str, Any]


class ModelInfo(BaseModel):
    """模型資訊"""
    filename: str
    model_type: str
    symbol: str
    file_size: int
    file_hash: str
    uploaded_at: str
    loaded: bool


class ModelsListResponse(BaseModel):
    """模型列表響應"""
    models: List[ModelInfo]
    total: int


class ModelReloadResponse(BaseModel):
    """模型重載響應"""
    success: bool
    message: str
    loaded_models: List[str]


# ==================== 輔助函數 ====================

def calculate_file_hash(file_path: Path) -> str:
    """計算檔案 MD5 hash"""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_model_metadata(model_path: Path) -> Dict[str, Any]:
    """獲取模型元數據"""
    metadata_path = model_path.with_suffix('.json')

    if metadata_path.exists():
        with open(metadata_path) as f:
            return json.load(f)

    # 從檔名解析基本資訊
    name = model_path.stem
    parts = name.split('_')

    return {
        "model_type": parts[0] if parts else "unknown",
        "symbol": parts[1] if len(parts) > 1 else "unknown",
        "uploaded_at": datetime.fromtimestamp(model_path.stat().st_mtime).isoformat()
    }


def get_all_models() -> List[Dict[str, Any]]:
    """獲取所有模型資訊"""
    models = []

    for ext in [".joblib", ".pt", ".pkl", ".h5"]:
        for model_path in MODEL_DIR.glob(f"*{ext}"):
            metadata = get_model_metadata(model_path)
            models.append({
                "filename": model_path.name,
                "model_type": metadata.get("model_type", "unknown"),
                "symbol": metadata.get("symbol", "unknown"),
                "file_size": model_path.stat().st_size,
                "file_hash": calculate_file_hash(model_path),
                "uploaded_at": metadata.get("uploaded_at",
                    datetime.fromtimestamp(model_path.stat().st_mtime).isoformat()),
                "loaded": model_path.name in _loaded_models
            })

    # 按上傳時間排序
    models.sort(key=lambda x: x["uploaded_at"], reverse=True)
    return models


async def load_model(model_path: Path) -> Any:
    """載入模型到記憶體"""
    import joblib

    ext = model_path.suffix.lower()

    try:
        if ext == ".joblib":
            model = joblib.load(model_path)
        elif ext == ".pt":
            import torch
            model = torch.load(model_path, map_location="cpu")
        elif ext == ".pkl":
            import pickle
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        elif ext == ".h5":
            try:
                from tensorflow import keras
                model = keras.models.load_model(model_path)
            except ImportError:
                logger.warning("TensorFlow not installed, skipping .h5 model")
                return None
        else:
            logger.warning(f"Unknown model format: {ext}")
            return None

        _loaded_models[model_path.name] = model
        logger.info(f"Loaded model: {model_path.name}")
        return model

    except Exception as e:
        logger.error(f"Failed to load model {model_path.name}: {e}")
        return None


async def reload_all_models():
    """重新載入所有模型"""
    global _loaded_models
    _loaded_models = {}

    loaded = []

    for ext in [".joblib", ".pt", ".pkl", ".h5"]:
        for model_path in MODEL_DIR.glob(f"*{ext}"):
            model = await load_model(model_path)
            if model is not None:
                loaded.append(model_path.name)

    logger.info(f"Loaded {len(loaded)} models: {loaded}")
    return loaded


# ==================== API 端點 ====================

@router.get("", response_model=ModelsListResponse)
async def list_models():
    """列出所有模型（無需認證）"""
    models = get_all_models()
    return ModelsListResponse(
        models=[ModelInfo(**m) for m in models],
        total=len(models)
    )


@router.post("/upload", response_model=ModelUploadResponse)
async def upload_model(
    request: ModelUploadRequest,
    background_tasks: BackgroundTasks,
    _: str = Depends(verify_api_key)
):
    """上傳模型

    接收 Base64 編碼的模型檔案並保存。
    """
    # 驗證檔名
    if not request.filename or ".." in request.filename or "/" in request.filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # 驗證副檔名
    valid_extensions = [".joblib", ".pt", ".pkl", ".h5"]
    ext = Path(request.filename).suffix.lower()
    if ext not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension. Allowed: {valid_extensions}"
        )

    model_path = MODEL_DIR / request.filename

    # 檢查是否已存在
    if model_path.exists() and not request.force:
        raise HTTPException(
            status_code=409,
            detail="Model already exists. Use force=true to overwrite."
        )

    try:
        # 解碼 Base64 數據
        model_data = base64.b64decode(request.data)

        # 保存模型
        with open(model_path, "wb") as f:
            f.write(model_data)

        file_hash = calculate_file_hash(model_path)
        uploaded_at = datetime.now().isoformat()

        # 保存元數據
        if request.metadata:
            metadata = request.metadata.copy()
            metadata["file_hash"] = file_hash
            metadata["uploaded_at"] = uploaded_at

            metadata_path = model_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        logger.info(f"Model uploaded: {request.filename} ({len(model_data)} bytes)")

        # 在背景載入模型
        background_tasks.add_task(load_model, model_path)

        return ModelUploadResponse(
            success=True,
            message="Model uploaded successfully",
            filename=request.filename,
            file_hash=file_hash,
            uploaded_at=uploaded_at
        )

    except base64.binascii.Error:
        raise HTTPException(status_code=400, detail="Invalid Base64 data")
    except Exception as e:
        logger.error(f"Failed to upload model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metadata")
async def upload_metadata(
    request: ModelMetadataRequest,
    _: str = Depends(verify_api_key)
):
    """上傳模型元數據"""
    # 驗證檔名
    if not request.filename or ".." in request.filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # 確保副檔名是 .json
    filename = request.filename
    if not filename.endswith('.json'):
        filename = Path(filename).stem + '.json'

    metadata_path = MODEL_DIR / filename

    try:
        with open(metadata_path, 'w') as f:
            json.dump(request.metadata, f, indent=2)

        return {"success": True, "message": "Metadata saved"}

    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reload", response_model=ModelReloadResponse)
async def reload_models():
    """重新載入所有模型（無需認證）

    用於更新模型後，讓交易引擎使用新模型。
    """
    try:
        loaded = await reload_all_models()

        return ModelReloadResponse(
            success=True,
            message=f"Reloaded {len(loaded)} models",
            loaded_models=loaded
        )

    except Exception as e:
        logger.error(f"Failed to reload models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/loaded/list")
async def get_loaded_models():
    """獲取已載入到記憶體的模型列表（無需認證）"""
    return {
        "loaded_models": list(_loaded_models.keys()),
        "total": len(_loaded_models)
    }


@router.get("/{filename}")
async def get_model_info(
    filename: str
):
    """獲取指定模型的資訊"""
    model_path = MODEL_DIR / filename

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    metadata = get_model_metadata(model_path)

    return {
        "filename": filename,
        "model_type": metadata.get("model_type", "unknown"),
        "symbol": metadata.get("symbol", "unknown"),
        "file_size": model_path.stat().st_size,
        "file_hash": calculate_file_hash(model_path),
        "uploaded_at": metadata.get("uploaded_at"),
        "loaded": filename in _loaded_models,
        "metadata": metadata
    }


@router.delete("/{filename}")
async def delete_model(
    filename: str,
    _: str = Depends(verify_api_key)
):
    """刪除模型"""
    model_path = MODEL_DIR / filename

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        # 從記憶體移除
        if filename in _loaded_models:
            del _loaded_models[filename]

        # 刪除檔案
        model_path.unlink()

        # 刪除元數據
        metadata_path = model_path.with_suffix('.json')
        if metadata_path.exists():
            metadata_path.unlink()

        logger.info(f"Model deleted: {filename}")

        return {"success": True, "message": f"Deleted {filename}"}

    except Exception as e:
        logger.error(f"Failed to delete model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 獲取模型實例的函數 ====================

def get_model(model_name: str) -> Optional[Any]:
    """獲取已載入的模型實例

    Args:
        model_name: 模型檔名

    Returns:
        模型實例，如果未載入則返回 None
    """
    return _loaded_models.get(model_name)


def get_latest_model(model_type: str = None) -> Optional[Any]:
    """獲取最新載入的模型

    Args:
        model_type: 模型類型過濾

    Returns:
        模型實例
    """
    if not _loaded_models:
        return None

    if model_type:
        matching = [
            (name, model) for name, model in _loaded_models.items()
            if model_type in name
        ]
        if matching:
            return matching[-1][1]
        return None

    return list(_loaded_models.values())[-1]


# ==================== iOS 本地訓練模型 API ====================

@ml_router.post("/upload", response_model=MLModelUploadResponse)
async def upload_ios_trained_model(
    request: MLModelUploadRequest,
    _: str = Depends(verify_api_key)
):
    """上傳 iOS 本地訓練的模型

    接收從 iOS App 本地訓練後推送的模型數據。
    模型權重和特徵重要性會被保存，用於後續交易決策。
    """
    try:
        import uuid

        # 生成唯一模型 ID
        model_id = f"ios_{request.symbol}_{request.model_type}_{uuid.uuid4().hex[:8]}"
        registered_at = datetime.now().isoformat()

        # 構建模型數據
        model_data = {
            "model_id": model_id,
            "symbol": request.symbol,
            "model_type": request.model_type,
            "metrics": {
                "accuracy": request.accuracy,
                "precision": request.precision,
                "recall": request.recall,
                "f1_score": request.f1_score
            },
            "weights": request.weights,
            "feature_importance": request.feature_importance,
            "trained_at": request.trained_at,
            "registered_at": registered_at,
            "data_points": request.data_points,
            "source": "ios_local_training"
        }

        # 保存到快取
        _ios_trained_models[model_id] = model_data

        # 保存到本地文件
        ios_models_dir = MODEL_DIR / "ios_trained"
        ios_models_dir.mkdir(parents=True, exist_ok=True)

        model_file = ios_models_dir / f"{model_id}.json"
        with open(model_file, 'w') as f:
            json.dump(model_data, f, indent=2)

        logger.info(f"iOS trained model uploaded: {model_id} for {request.symbol} ({request.model_type})")
        logger.info(f"  Metrics: accuracy={request.accuracy:.4f}, f1={request.f1_score:.4f}")
        logger.info(f"  Data points: {request.data_points}, Weights: {len(request.weights)}")

        return MLModelUploadResponse(
            success=True,
            message=f"Model {model_id} registered successfully",
            model_id=model_id,
            registered_at=registered_at
        )

    except Exception as e:
        logger.error(f"Failed to upload iOS trained model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@ml_router.get("")
async def list_ios_trained_models():
    """列出所有 iOS 本地訓練的模型"""
    # 從文件加載所有已保存的模型
    ios_models_dir = MODEL_DIR / "ios_trained"
    models = []

    if ios_models_dir.exists():
        for model_file in ios_models_dir.glob("*.json"):
            try:
                with open(model_file) as f:
                    model_data = json.load(f)
                    models.append({
                        "model_id": model_data.get("model_id"),
                        "symbol": model_data.get("symbol"),
                        "model_type": model_data.get("model_type"),
                        "accuracy": model_data.get("metrics", {}).get("accuracy"),
                        "f1_score": model_data.get("metrics", {}).get("f1_score"),
                        "trained_at": model_data.get("trained_at"),
                        "registered_at": model_data.get("registered_at"),
                        "data_points": model_data.get("data_points")
                    })
            except Exception as e:
                logger.warning(f"Failed to load model file {model_file}: {e}")

    # 按註冊時間排序
    models.sort(key=lambda x: x.get("registered_at", ""), reverse=True)

    return {
        "models": models,
        "total": len(models)
    }


# ==================== iOS 模型上傳到 Arena 競技場 ====================
# 注意：這些靜態路徑必須在 /{model_id} 動態路由之前定義

class IOSModelToArenaRequest(BaseModel):
    """iOS 模型註冊到 Arena 請求"""
    model_id: str  # iOS 訓練的模型 ID
    as_challenger: bool = True  # 是否作為挑戰者


class IOSModelToArenaResponse(BaseModel):
    """iOS 模型註冊到 Arena 響應"""
    success: bool
    message: str
    arena_version: str
    model_id: str


@ml_router.get("/arena-compatible")
async def list_arena_compatible_models():
    """列出可以註冊到 Arena 的 iOS 訓練模型

    返回所有準確率 > 50% 的模型，這些模型可以被推送到 Arena 進行比較。
    """
    ios_models_dir = MODEL_DIR / "ios_trained"
    compatible_models = []

    if ios_models_dir.exists():
        for model_file in ios_models_dir.glob("*.json"):
            try:
                with open(model_file) as f:
                    model_data = json.load(f)
                    accuracy = model_data.get("metrics", {}).get("accuracy", 0)

                    # 只返回準確率 > 50% 的模型
                    if accuracy > 0.5:
                        compatible_models.append({
                            "model_id": model_data.get("model_id"),
                            "symbol": model_data.get("symbol"),
                            "model_type": model_data.get("model_type"),
                            "accuracy": accuracy,
                            "f1_score": model_data.get("metrics", {}).get("f1_score"),
                            "data_points": model_data.get("data_points"),
                            "trained_at": model_data.get("trained_at"),
                            "arena_version_preview": f"ios-{model_data.get('symbol', 'unknown').lower()}-{model_data.get('model_type', 'unknown').lower()}-{int(accuracy * 100)}"
                        })
            except Exception as e:
                logger.warning(f"Failed to load model file {model_file}: {e}")

    # 按準確率排序
    compatible_models.sort(key=lambda x: x.get("accuracy", 0), reverse=True)

    return {
        "models": compatible_models,
        "total": len(compatible_models),
        "min_accuracy_threshold": 0.5
    }


@ml_router.post("/to-arena", response_model=IOSModelToArenaResponse)
async def register_ios_model_to_arena(
    request: IOSModelToArenaRequest,
    _: str = Depends(verify_api_key)
):
    """將 iOS 訓練的模型註冊到 Arena 競技場

    這個端點允許將本地訓練的 iOS 模型推送到 Arena 系統，
    讓它可以作為挑戰者與當前冠軍進行影子交易比較。
    需要 API Key 認證。
    """
    try:
        from src.services.arena_service import arena_service

        # 獲取 iOS 模型數據
        model_file = MODEL_DIR / "ios_trained" / f"{request.model_id}.json"
        if not model_file.exists():
            raise HTTPException(status_code=404, detail=f"iOS model {request.model_id} not found")

        with open(model_file) as f:
            ios_model = json.load(f)

        # 生成 Arena 版本號
        symbol = ios_model.get("symbol", "UNKNOWN")
        model_type = ios_model.get("model_type", "unknown")
        accuracy = ios_model.get("metrics", {}).get("accuracy", 0)

        # 格式: ios-{symbol}-{type}-{accuracy}
        arena_version = f"ios-{symbol.lower()}-{model_type.lower()}-{int(accuracy * 100)}"

        # 構建模型配置（包含 iOS 模型的元數據）
        config = {
            "source": "ios_local_training",
            "ios_model_id": request.model_id,
            "symbol": symbol,
            "accuracy": accuracy,
            "f1_score": ios_model.get("metrics", {}).get("f1_score", 0),
            "data_points": ios_model.get("data_points", 0),
            "trained_at": ios_model.get("trained_at"),
            "feature_importance": ios_model.get("feature_importance", {}),
            "weights": ios_model.get("weights", [])
        }

        # 註冊到 Arena
        success = await arena_service.register_model(
            version=arena_version,
            model_type=f"iOS-{model_type}",
            config=config
        )

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to register model to Arena"
            )

        # 如果作為挑戰者，啟動影子交易
        if request.as_challenger:
            await arena_service.start_shadow_trading(arena_version)

        logger.info(f"iOS model {request.model_id} registered to Arena as {arena_version}")

        return IOSModelToArenaResponse(
            success=True,
            message=f"Model registered to Arena as {arena_version}",
            arena_version=arena_version,
            model_id=request.model_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to register iOS model to Arena: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 策略組合 API（必須在動態路由之前）====================

from src.services.strategy_ensemble_service import (
    ensemble_service,
    EnsembleMethod,
    ModelPrediction,
    EnsemblePrediction
)


class EnsemblePredictionRequest(BaseModel):
    """策略組合預測請求"""
    symbol: str
    features: Dict[str, float]
    method: str = "confidence_weighted"  # voting, weighted_average, confidence_weighted, best_model
    model_ids: Optional[List[str]] = None  # 指定模型列表，為空則使用所有可用模型
    min_models: int = 2


class EnsemblePredictionResponse(BaseModel):
    """策略組合預測響應"""
    symbol: str
    final_signal: int
    final_confidence: float
    final_strength: float
    recommendation: str
    reasoning: str
    method_used: str
    participating_models: int
    agreement_ratio: float
    buy_votes: int
    sell_votes: int
    hold_votes: int
    individual_predictions: List[Dict[str, Any]]
    ensemble_metrics: Dict[str, Any]
    timestamp: str


@ml_router.get("/ensemble/methods")
async def list_ensemble_methods():
    """列出可用的組合方法及其說明"""
    return {
        "methods": [
            {
                "id": "voting",
                "name": "投票法",
                "description": "多數決定：選擇獲得最多票數的信號方向"
            },
            {
                "id": "weighted_average",
                "name": "加權平均",
                "description": "根據模型歷史準確率對信號進行加權平均"
            },
            {
                "id": "confidence_weighted",
                "name": "信心加權",
                "description": "根據每個模型的信心度和準確率組合加權（推薦）"
            },
            {
                "id": "best_model",
                "name": "最佳模型",
                "description": "直接使用準確率和信心度最高的單一模型"
            }
        ],
        "default": "confidence_weighted",
        "recommendation": "對於一般使用場景，建議使用 confidence_weighted 方法，它綜合考慮了模型準確率和當前預測的信心度。"
    }


@ml_router.post("/ensemble/set-weight")
async def set_model_weight(
    model_id: str,
    weight: float,
    _: str = Depends(verify_api_key)
):
    """設置模型在組合中的權重

    手動調整特定模型的權重（覆蓋默認的準確率計算）。
    """
    if weight < 0 or weight > 1:
        raise HTTPException(status_code=400, detail="Weight must be between 0 and 1")

    ensemble_service.set_model_weight(model_id, weight)

    return {
        "success": True,
        "model_id": model_id,
        "weight": weight,
        "message": f"Weight for {model_id} set to {weight}"
    }


# ==================== 動態路由（必須放在靜態路由之後）====================

@ml_router.get("/{model_id}")
async def get_ios_trained_model(model_id: str):
    """獲取指定 iOS 訓練模型的詳細資訊"""
    # 先檢查快取
    if model_id in _ios_trained_models:
        return _ios_trained_models[model_id]

    # 從文件加載
    model_file = MODEL_DIR / "ios_trained" / f"{model_id}.json"
    if model_file.exists():
        with open(model_file) as f:
            return json.load(f)

    raise HTTPException(status_code=404, detail="Model not found")


@ml_router.delete("/{model_id}")
async def delete_ios_trained_model(
    model_id: str,
    _: str = Depends(verify_api_key)
):
    """刪除指定 iOS 訓練模型"""
    # 從快取移除
    if model_id in _ios_trained_models:
        del _ios_trained_models[model_id]

    # 刪除文件
    model_file = MODEL_DIR / "ios_trained" / f"{model_id}.json"
    if model_file.exists():
        model_file.unlink()
        logger.info(f"iOS trained model deleted: {model_id}")
        return {"success": True, "message": f"Deleted {model_id}"}

    raise HTTPException(status_code=404, detail="Model not found")


# ==================== 獲取最佳 iOS 模型的函數 ====================

def get_best_ios_model(symbol: str = None, model_type: str = None) -> Optional[Dict[str, Any]]:
    """獲取最佳 iOS 訓練模型

    Args:
        symbol: 交易對過濾
        model_type: 模型類型過濾

    Returns:
        最佳模型數據（按 F1 分數排序）
    """
    ios_models_dir = MODEL_DIR / "ios_trained"
    if not ios_models_dir.exists():
        return None

    best_model = None
    best_f1 = 0.0

    for model_file in ios_models_dir.glob("*.json"):
        try:
            with open(model_file) as f:
                model_data = json.load(f)

                # 過濾條件
                if symbol and model_data.get("symbol") != symbol:
                    continue
                if model_type and model_data.get("model_type") != model_type:
                    continue

                # 比較 F1 分數
                f1 = model_data.get("metrics", {}).get("f1_score", 0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model_data

        except Exception as e:
            logger.warning(f"Failed to load model file {model_file}: {e}")

    return best_model


# ==================== iOS 模型預測 API ====================

class PredictionRequest(BaseModel):
    """預測請求"""
    symbol: str
    features: Dict[str, float]  # 特徵數據
    model_id: Optional[str] = None  # 指定模型，若為空則使用最佳模型


class PredictionResponse(BaseModel):
    """預測響應"""
    symbol: str
    model_id: str
    model_type: str
    prediction: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0.0 - 1.0
    signal_strength: float  # 訊號強度
    feature_contributions: Dict[str, float]  # 各特徵的貢獻度
    timestamp: str


def calculate_dynamic_thresholds(
    features: Dict[str, float],
    base_threshold: float = 0.15
) -> tuple:
    """計算動態買賣閾值

    根據市場波動率動態調整閾值：
    - 高波動率 → 更寬的閾值（避免假訊號）
    - 低波動率 → 更緊的閾值（捕捉小波動）

    Args:
        features: 市場特徵數據
        base_threshold: 基礎閾值（默認 0.15）

    Returns:
        (buy_threshold, sell_threshold) 動態閾值元組
    """
    # 提取波動率相關特徵
    volatility = None
    volatility_keys = ['volatility', 'atr', 'vol5', 'vol10', 'vol20', 'high_low_range']

    for key in volatility_keys:
        for feature_key, feature_value in features.items():
            if key in feature_key.lower():
                volatility = feature_value
                break
        if volatility is not None:
            break

    # 提取成交量比率（用於補充波動率判斷）
    volume_ratio = None
    for feature_key, feature_value in features.items():
        if 'volume' in feature_key.lower() and 'ratio' in feature_key.lower():
            volume_ratio = feature_value
            break

    # 計算動態調整係數
    volatility_multiplier = 1.0

    if volatility is not None:
        # 波動率正規化（假設正常波動率在 0.01-0.05 範圍）
        # 根據波動率調整閾值：
        # - 波動率 < 0.01 → 縮小閾值（低波動市場，需要更敏感）
        # - 波動率 > 0.03 → 擴大閾值（高波動市場，需要更保守）
        if volatility < 0.01:
            volatility_multiplier = 0.7  # 低波動，更敏感
        elif volatility < 0.02:
            volatility_multiplier = 0.85  # 正常偏低
        elif volatility < 0.03:
            volatility_multiplier = 1.0   # 正常
        elif volatility < 0.05:
            volatility_multiplier = 1.2   # 高波動
        else:
            volatility_multiplier = 1.5   # 極高波動

    # 成交量調整（高成交量通常意味著更可靠的訊號）
    volume_multiplier = 1.0
    if volume_ratio is not None:
        if volume_ratio > 2.0:
            volume_multiplier = 0.9   # 放量，可以更敏感
        elif volume_ratio < 0.5:
            volume_multiplier = 1.15  # 縮量，需要更保守

    # 計算最終閾值
    final_multiplier = volatility_multiplier * volume_multiplier

    # 閾值範圍限制：0.08 到 0.25
    adjusted_threshold = max(0.08, min(0.25, base_threshold * final_multiplier))

    buy_threshold = adjusted_threshold
    sell_threshold = -adjusted_threshold

    return buy_threshold, sell_threshold


def calculate_ios_model_prediction(
    model_data: Dict[str, Any],
    features: Dict[str, float]
) -> Dict[str, Any]:
    """使用 iOS 訓練的模型進行預測

    iOS 模型使用簡單的線性加權方式：
    - 根據 feature_importance 對特徵進行加權
    - 使用 weights 作為特徵權重
    - 計算加權分數來決定買賣訊號
    - 使用動態閾值根據市場波動率調整敏感度

    Args:
        model_data: iOS 訓練的模型數據
        features: 當前市場特徵

    Returns:
        預測結果
    """
    weights = model_data.get("weights", [])
    feature_importance = model_data.get("feature_importance", {})

    # 標準化特徵名稱映射
    feature_mapping = {
        "rsi": "rsi",
        "rsi_14": "rsi",
        "macd": "macd",
        "macd_line": "macd",
        "macd_signal": "macdSignal",
        "macd_histogram": "macdHistogram",
        "bb_position": "bbPosition",
        "bollinger_position": "bbPosition",
        "volume_ratio": "volumeRatio",
        "volume_change": "volumeRatio",
        "price_change": "priceChange",
        "price_change_24h": "priceChange",
        "ema_cross": "emaCross",
        "ema_signal": "emaCross",
        "volatility": "volatility",
        "atr": "volatility",
        "momentum": "momentum",
        "roc": "momentum",
    }

    # 計算加權分數
    weighted_score = 0.0
    total_weight = 0.0
    feature_contributions = {}

    # 使用 feature_importance 進行加權
    for raw_feature, importance in feature_importance.items():
        # 標準化特徵名稱
        feature_key = feature_mapping.get(raw_feature.lower(), raw_feature.lower())

        # 查找對應的特徵值
        feature_value = None
        for input_key, input_value in features.items():
            normalized_input = feature_mapping.get(input_key.lower(), input_key.lower())
            if normalized_input == feature_key or input_key.lower() == raw_feature.lower():
                feature_value = input_value
                break

        if feature_value is not None:
            # 正規化特徵值到 [-1, 1] 範圍
            normalized_value = normalize_feature(feature_key, feature_value)
            contribution = normalized_value * importance
            weighted_score += contribution
            total_weight += abs(importance)
            feature_contributions[raw_feature] = round(contribution, 4)

    # 計算最終分數 (-1 到 1)
    if total_weight > 0:
        final_score = weighted_score / total_weight
    else:
        final_score = 0.0

    # 決定預測結果
    # 使用動態閾值（根據市場波動率調整）
    buy_threshold, sell_threshold = calculate_dynamic_thresholds(features)

    if final_score > buy_threshold:
        prediction = "BUY"
        confidence = min(1.0, (final_score - buy_threshold) / (1.0 - buy_threshold) * 0.5 + 0.5)
    elif final_score < sell_threshold:
        prediction = "SELL"
        confidence = min(1.0, (abs(final_score) - abs(sell_threshold)) / (1.0 - abs(sell_threshold)) * 0.5 + 0.5)
    else:
        prediction = "HOLD"
        confidence = 1.0 - abs(final_score) / max(abs(buy_threshold), abs(sell_threshold))

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "signal_strength": round(abs(final_score), 4),
        "feature_contributions": feature_contributions,
        "raw_score": round(final_score, 4),
        "dynamic_thresholds": {
            "buy": round(buy_threshold, 4),
            "sell": round(sell_threshold, 4)
        }
    }


def normalize_feature(feature_name: str, value: float) -> float:
    """正規化特徵值到 [-1, 1] 範圍

    Args:
        feature_name: 特徵名稱
        value: 原始值

    Returns:
        正規化後的值
    """
    # 根據不同特徵的典型範圍進行正規化
    if feature_name == "rsi":
        # RSI: 0-100, 50 為中性
        return (value - 50) / 50
    elif feature_name == "macd":
        # MACD: 通常在 -100 到 100 之間
        return max(-1, min(1, value / 50))
    elif feature_name == "bbPosition":
        # BB Position: 0-1, 0.5 為中性
        return (value - 0.5) * 2
    elif feature_name == "volumeRatio":
        # Volume Ratio: 通常 0.5-2.0, 1.0 為中性
        return max(-1, min(1, (value - 1) * 2))
    elif feature_name == "priceChange":
        # Price Change: 百分比，通常 -10 到 10
        return max(-1, min(1, value / 5))
    elif feature_name == "emaCross":
        # EMA Cross: 已經是 -1 到 1
        return max(-1, min(1, value))
    elif feature_name == "volatility":
        # Volatility: 0-100, 高波動為負面
        return -max(-1, min(1, (value - 20) / 30))
    elif feature_name == "momentum":
        # Momentum: 通常 -20 到 20
        return max(-1, min(1, value / 10))
    else:
        # 默認：假設已經在合理範圍
        return max(-1, min(1, value))


@ml_router.post("/predict", response_model=PredictionResponse)
async def predict_with_ios_model(request: PredictionRequest):
    """使用 iOS 訓練的模型進行預測

    傳入當前市場特徵，返回買賣預測。
    如果未指定 model_id，將自動選擇該交易對的最佳模型。
    """
    try:
        # 獲取模型
        if request.model_id:
            # 使用指定模型
            model_file = MODEL_DIR / "ios_trained" / f"{request.model_id}.json"
            if not model_file.exists():
                raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
            with open(model_file) as f:
                model_data = json.load(f)
        else:
            # 使用最佳模型
            model_data = get_best_ios_model(symbol=request.symbol)
            if not model_data:
                raise HTTPException(
                    status_code=404,
                    detail=f"No trained model found for {request.symbol}"
                )

        # 進行預測
        prediction_result = calculate_ios_model_prediction(model_data, request.features)

        logger.info(f"Prediction for {request.symbol}: {prediction_result['prediction']} "
                   f"(confidence={prediction_result['confidence']:.2%}, "
                   f"strength={prediction_result['signal_strength']:.2%})")

        return PredictionResponse(
            symbol=request.symbol,
            model_id=model_data["model_id"],
            model_type=model_data["model_type"],
            prediction=prediction_result["prediction"],
            confidence=prediction_result["confidence"],
            signal_strength=prediction_result["signal_strength"],
            feature_contributions=prediction_result["feature_contributions"],
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@ml_router.get("/predict/{symbol}")
async def get_current_prediction(symbol: str):
    """獲取指定交易對的當前預測

    自動獲取最新市場數據並使用最佳模型進行預測。
    """
    try:
        from src.services.binance_client import BinanceClient

        # 獲取最佳模型
        model_data = get_best_ios_model(symbol=symbol)
        if not model_data:
            return {
                "symbol": symbol,
                "error": "No trained model available",
                "suggestion": "Please train a model for this symbol first"
            }

        # 嘗試獲取實時市場數據
        try:
            client = BinanceClient()

            # 獲取技術指標
            ticker = await client.get_ticker(symbol)
            klines = await client.get_klines(symbol, "1h", limit=50)

            # 計算特徵
            features = calculate_market_features(ticker, klines)

        except Exception as e:
            logger.warning(f"Failed to get real market data: {e}, using mock data")
            # 使用模擬數據
            features = {
                "rsi": 50.0,
                "macd": 0.0,
                "bbPosition": 0.5,
                "volumeRatio": 1.0,
                "priceChange": 0.0,
                "emaCross": 0.0,
                "volatility": 20.0,
                "momentum": 0.0
            }

        # 進行預測
        prediction_result = calculate_ios_model_prediction(model_data, features)

        return {
            "symbol": symbol,
            "model_id": model_data["model_id"],
            "model_type": model_data["model_type"],
            "model_accuracy": model_data.get("metrics", {}).get("accuracy", 0),
            "prediction": prediction_result["prediction"],
            "confidence": prediction_result["confidence"],
            "signal_strength": prediction_result["signal_strength"],
            "features_used": features,
            "feature_contributions": prediction_result["feature_contributions"],
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get current prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def calculate_market_features(ticker: Dict, klines: List) -> Dict[str, float]:
    """從市場數據計算特徵

    Args:
        ticker: 當前價格數據
        klines: K線數據

    Returns:
        特徵字典
    """
    import numpy as np

    # 提取收盤價
    closes = [float(k[4]) for k in klines]
    volumes = [float(k[5]) for k in klines]
    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]

    # 計算 RSI
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-14:])
    avg_loss = np.mean(losses[-14:])
    rs = avg_gain / avg_loss if avg_loss > 0 else 100
    rsi = 100 - (100 / (1 + rs))

    # 計算 MACD
    ema12 = np.mean(closes[-12:])
    ema26 = np.mean(closes[-26:])
    macd = ema12 - ema26

    # 計算布林帶位置
    sma20 = np.mean(closes[-20:])
    std20 = np.std(closes[-20:])
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20
    current_price = closes[-1]
    bb_position = (current_price - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0.5

    # 計算成交量比率
    avg_volume = np.mean(volumes[-20:])
    volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0

    # 計算價格變化
    price_change = ((closes[-1] - closes[-24]) / closes[-24] * 100) if len(closes) >= 24 else 0

    # 計算 EMA 交叉
    ema9 = np.mean(closes[-9:])
    ema21 = np.mean(closes[-21:])
    ema_cross = (ema9 - ema21) / ema21 * 100 if ema21 > 0 else 0

    # 計算波動率 (ATR)
    tr_list = []
    for i in range(1, min(14, len(closes))):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        tr_list.append(tr)
    atr = np.mean(tr_list) if tr_list else 0
    volatility = (atr / current_price * 100) if current_price > 0 else 0

    # 計算動量
    momentum = ((closes[-1] - closes[-10]) / closes[-10] * 100) if len(closes) >= 10 else 0

    return {
        "rsi": round(rsi, 2),
        "macd": round(macd, 4),
        "bbPosition": round(bb_position, 4),
        "volumeRatio": round(volume_ratio, 4),
        "priceChange": round(price_change, 4),
        "emaCross": round(ema_cross, 4),
        "volatility": round(volatility, 4),
        "momentum": round(momentum, 4)
    }


# ==================== 策略組合預測路由（需要 calculate_ios_model_prediction）====================

@ml_router.post("/ensemble/predict", response_model=EnsemblePredictionResponse)
async def ensemble_prediction(request: EnsemblePredictionRequest):
    """使用多個模型進行組合預測

    組合多個 iOS 訓練模型的預測結果，產生更可靠的交易信號。
    支持多種組合方法：投票法、加權平均、信心加權、最佳模型。
    """
    try:
        # 獲取所有可用模型
        ios_models_dir = MODEL_DIR / "ios_trained"
        available_models = []

        if ios_models_dir.exists():
            for model_file in ios_models_dir.glob("*.json"):
                try:
                    with open(model_file) as f:
                        model_data = json.load(f)
                        # 過濾：只選擇匹配交易對的模型
                        if model_data.get("symbol") == request.symbol:
                            # 如果指定了模型列表，檢查是否在列表中
                            if request.model_ids is None or model_data.get("model_id") in request.model_ids:
                                available_models.append(model_data)
                except Exception as e:
                    logger.warning(f"Failed to load model {model_file}: {e}")

        if not available_models:
            raise HTTPException(
                status_code=404,
                detail=f"No trained models found for {request.symbol}"
            )

        # 對每個模型進行預測
        predictions = []
        for model_data in available_models:
            try:
                result = calculate_ios_model_prediction(model_data, request.features)

                # 轉換預測結果為信號
                signal = 0
                if result["prediction"] == "BUY":
                    signal = 1
                elif result["prediction"] == "SELL":
                    signal = -1

                predictions.append(ModelPrediction(
                    model_id=model_data["model_id"],
                    model_type=model_data["model_type"],
                    signal=signal,
                    confidence=result["confidence"],
                    strength=result["signal_strength"],
                    accuracy=model_data.get("metrics", {}).get("accuracy", 0.5),
                    features_used=len(model_data.get("feature_importance", {}))
                ))
            except Exception as e:
                logger.warning(f"Model {model_data.get('model_id')} prediction failed: {e}")

        if not predictions:
            raise HTTPException(
                status_code=500,
                detail="All model predictions failed"
            )

        # 選擇組合方法
        method_map = {
            "voting": EnsembleMethod.VOTING,
            "weighted_average": EnsembleMethod.WEIGHTED_AVG,
            "confidence_weighted": EnsembleMethod.CONFIDENCE_WEIGHTED,
            "best_model": EnsembleMethod.BEST_MODEL
        }
        method = method_map.get(request.method, EnsembleMethod.CONFIDENCE_WEIGHTED)

        # 執行組合預測
        ensemble_result = ensemble_service.combine_predictions(
            predictions=predictions,
            method=method,
            min_models=request.min_models
        )

        # 計算組合指標
        ensemble_metrics = ensemble_service.calculate_ensemble_metrics(predictions)

        logger.info(f"Ensemble prediction for {request.symbol}: {ensemble_result.recommendation} "
                   f"(confidence={ensemble_result.final_confidence:.2%}, "
                   f"agreement={ensemble_result.agreement_ratio:.2%}, "
                   f"models={ensemble_result.participating_models})")

        return EnsemblePredictionResponse(
            symbol=request.symbol,
            final_signal=ensemble_result.final_signal,
            final_confidence=ensemble_result.final_confidence,
            final_strength=ensemble_result.final_strength,
            recommendation=ensemble_result.recommendation,
            reasoning=ensemble_result.reasoning,
            method_used=ensemble_result.method_used,
            participating_models=ensemble_result.participating_models,
            agreement_ratio=ensemble_result.agreement_ratio,
            buy_votes=ensemble_result.buy_votes,
            sell_votes=ensemble_result.sell_votes,
            hold_votes=ensemble_result.hold_votes,
            individual_predictions=ensemble_result.individual_predictions,
            ensemble_metrics=ensemble_metrics,
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ensemble prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@ml_router.get("/ensemble/predict/{symbol}")
async def get_ensemble_prediction(
    symbol: str,
    method: str = "confidence_weighted"
):
    """獲取指定交易對的組合預測

    自動獲取最新市場數據，使用所有可用模型進行組合預測。
    """
    try:
        from src.services.binance_client import BinanceClient

        # 嘗試獲取實時市場數據
        try:
            client = BinanceClient()
            ticker = await client.get_ticker(symbol)
            klines = await client.get_klines(symbol, "1h", limit=50)
            features = calculate_market_features(ticker, klines)
        except Exception as e:
            logger.warning(f"Failed to get real market data: {e}, using mock data")
            features = {
                "rsi": 50.0,
                "macd": 0.0,
                "bbPosition": 0.5,
                "volumeRatio": 1.0,
                "priceChange": 0.0,
                "emaCross": 0.0,
                "volatility": 20.0,
                "momentum": 0.0
            }

        # 使用組合預測
        request = EnsemblePredictionRequest(
            symbol=symbol,
            features=features,
            method=method
        )

        return await ensemble_prediction(request)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get ensemble prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 風險管理 API ====================

from src.services.risk_management_service import risk_service, RiskConfig


class RiskConfigRequest(BaseModel):
    """風險配置請求"""
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    max_hold_hours: Optional[int] = None
    min_trade_interval_minutes: Optional[int] = None
    max_trades_per_day: Optional[int] = None
    model_expiry_days: Optional[int] = None
    model_warning_days: Optional[int] = None


@ml_router.get("/risk/config")
async def get_risk_config():
    """獲取當前風險管理配置"""
    return {
        "config": risk_service.config.to_dict(),
        "description": {
            "stop_loss_pct": "停損百分比（觸發後自動平倉）",
            "take_profit_pct": "止盈百分比（觸發後自動平倉）",
            "trailing_stop_pct": "追蹤止損百分比（從高點回撤）",
            "max_drawdown_pct": "最大回撤百分比（觸發後暫停交易）",
            "max_hold_hours": "最大持倉時間（小時）",
            "min_trade_interval_minutes": "最小交易間隔（分鐘）",
            "max_trades_per_day": "每日最大交易次數",
            "model_expiry_days": "模型過期天數",
            "model_warning_days": "模型即將過期警告天數"
        }
    }


@ml_router.post("/risk/config")
async def update_risk_config(
    request: RiskConfigRequest,
    _: str = Depends(verify_api_key)
):
    """更新風險管理配置"""
    updates = {k: v for k, v in request.dict().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No configuration to update")

    risk_service.update_config(**updates)

    return {
        "success": True,
        "message": f"Updated {len(updates)} configuration(s)",
        "updated": updates,
        "current_config": risk_service.config.to_dict()
    }


@ml_router.get("/risk/alerts")
async def get_risk_alerts(
    model_version: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 50
):
    """獲取風險警報列表"""
    alerts = risk_service.get_recent_alerts(
        model_version=model_version,
        severity=severity,
        limit=limit
    )
    return {
        "count": len(alerts),
        "alerts": alerts,
        "filters": {
            "model_version": model_version,
            "severity": severity
        }
    }


@ml_router.post("/risk/check/{model_version}")
async def check_model_risk(
    model_version: str,
    _: str = Depends(verify_api_key)
):
    """手動檢查模型風險

    檢查指定模型的所有持倉風險和帳戶風險。
    """
    from src.services.arena_service import arena_service
    from decimal import Decimal

    # 獲取虛擬帳戶
    account = await arena_service.get_virtual_account(model_version)
    if not account:
        raise HTTPException(status_code=404, detail=f"Model {model_version} not found")

    # 獲取所有持倉
    positions = await arena_service.get_virtual_positions(model_version)

    risk_issues = []
    total_value = Decimal(str(account['total_value']))

    # 檢查每個持倉的風險
    for pos in positions:
        should_close, action, reason = await risk_service.check_position_risk(
            model_version=model_version,
            symbol=pos.symbol,
            entry_price=pos.entry_price,
            current_price=pos.current_price,
            quantity=pos.quantity,
            entry_time=datetime.utcnow(),  # 簡化：實際應該從數據庫獲取
            peak_price=pos.current_price  # 簡化
        )

        if should_close:
            risk_issues.append({
                "symbol": pos.symbol,
                "action": str(action) if action else None,
                "reason": reason,
                "recommended": "CLOSE_POSITION"
            })

    # 檢查帳戶風險
    should_stop, account_action, account_reason = await risk_service.check_account_risk(
        model_version=model_version,
        current_value=total_value
    )

    if should_stop:
        risk_issues.append({
            "symbol": None,
            "action": str(account_action) if account_action else None,
            "reason": account_reason,
            "recommended": "PAUSE_TRADING"
        })

    # 檢查交易頻率
    can_trade, freq_reason = await risk_service.can_trade(model_version, "")
    if not can_trade:
        risk_issues.append({
            "symbol": None,
            "action": "FREQUENCY_LIMIT",
            "reason": freq_reason,
            "recommended": "WAIT"
        })

    return {
        "model_version": model_version,
        "total_value": float(total_value),
        "positions_checked": len(positions),
        "risk_issues": risk_issues,
        "risk_level": "HIGH" if len(risk_issues) > 0 else "NORMAL",
        "timestamp": datetime.now().isoformat()
    }


@ml_router.post("/risk/auto-manage")
async def trigger_auto_risk_management(
    _: str = Depends(verify_api_key)
):
    """手動觸發自動風險管理

    檢查所有模型的持倉風險，自動平倉觸發風險的持倉。
    """
    try:
        actions = await risk_service.auto_manage_positions()

        return {
            "success": True,
            "actions_taken": len(actions),
            "details": actions,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Auto risk management failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
