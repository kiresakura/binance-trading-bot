#!/usr/bin/env python3
"""
模型同步腳本

將本地訓練好的模型上傳到 Railway 環境。
支持多種上傳方式：API 上傳、Volume 同步等。

使用方式：
    # 上傳最新模型
    python scripts/sync_models.py --upload-latest

    # 上傳指定模型
    python scripts/sync_models.py --upload models/ensemble_BTCUSDT_20241212.joblib

    # 列出本地模型
    python scripts/sync_models.py --list

    # 列出 Railway 上的模型
    python scripts/sync_models.py --list-remote

    # 通知 Railway 重新載入模型
    python scripts/sync_models.py --reload
"""

import argparse
import asyncio
import base64
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import aiohttp

# 添加專案根目錄到 Python 路徑
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import logger

# 從環境變數讀取配置（不提供任何預設網址或金鑰，避免洩漏正式環境資訊）
RAILWAY_API_URL = os.getenv("RAILWAY_API_URL", "")
RAILWAY_API_KEY = os.getenv("RAILWAY_API_KEY", "")
MODEL_DIR = Path(os.getenv("MODEL_OUTPUT_DIR", PROJECT_ROOT / "models"))


def _require_remote_config() -> None:
    """需要連線遠端時，確認必要環境變數已設定。"""
    missing = [
        name
        for name, value in (("RAILWAY_API_URL", RAILWAY_API_URL), ("RAILWAY_API_KEY", RAILWAY_API_KEY))
        if not value
    ]
    if missing:
        logger.error(f"❌ 缺少環境變數: {', '.join(missing)}（請於 .env.local 設定後再執行）")
        sys.exit(1)


def get_model_files(pattern: str = "*") -> List[Path]:
    """獲取本地模型檔案列表"""
    models = []

    for ext in [".joblib", ".pt", ".pkl", ".h5"]:
        models.extend(MODEL_DIR.glob(f"**/*{ext}"))

    # 按修改時間排序（最新在前）
    models.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    return models


def get_latest_model(model_type: str = None) -> Optional[Path]:
    """獲取最新的模型檔案"""
    models = get_model_files()

    if model_type:
        models = [m for m in models if model_type in m.name]

    return models[0] if models else None


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
        "trained_at": datetime.fromtimestamp(model_path.stat().st_mtime).isoformat(),
        "file_size": model_path.stat().st_size,
        "file_hash": calculate_file_hash(model_path)
    }


async def upload_model(model_path: Path, force: bool = False) -> bool:
    """上傳模型到 Railway

    Args:
        model_path: 模型檔案路徑
        force: 是否強制覆蓋

    Returns:
        是否上傳成功
    """
    if not model_path.exists():
        logger.error(f"❌ Model file not found: {model_path}")
        return False

    logger.info(f"📤 Uploading model: {model_path.name}")

    # 讀取模型檔案
    with open(model_path, "rb") as f:
        model_data = f.read()

    # Base64 編碼
    model_b64 = base64.b64encode(model_data).decode()

    # 獲取元數據
    metadata = get_model_metadata(model_path)
    metadata["filename"] = model_path.name
    metadata["file_hash"] = calculate_file_hash(model_path)

    # 構建請求
    payload = {
        "filename": model_path.name,
        "data": model_b64,
        "metadata": metadata,
        "force": force
    }

    # 上傳到 Railway
    headers = {
        "X-API-Key": RAILWAY_API_KEY,
        "Content-Type": "application/json"
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{RAILWAY_API_URL}/api/v1/models/upload",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=300)  # 5 分鐘超時
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"✅ Upload successful: {result.get('message', 'OK')}")

                    # 也上傳元數據檔案
                    metadata_path = model_path.with_suffix('.json')
                    if metadata_path.exists():
                        await upload_metadata(metadata_path, headers)

                    return True
                elif response.status == 409:
                    logger.warning("⚠️ Model already exists. Use --force to overwrite.")
                    return False
                else:
                    error = await response.text()
                    logger.error(f"❌ Upload failed ({response.status}): {error}")
                    return False

        except aiohttp.ClientError as e:
            logger.error(f"❌ Network error: {e}")
            return False
        except asyncio.TimeoutError:
            logger.error("❌ Upload timeout")
            return False


async def upload_metadata(metadata_path: Path, headers: dict) -> bool:
    """上傳模型元數據"""
    with open(metadata_path) as f:
        metadata = json.load(f)

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{RAILWAY_API_URL}/api/v1/models/metadata",
                json={"filename": metadata_path.name, "metadata": metadata},
                headers=headers
            ) as response:
                return response.status == 200
        except Exception as e:
            logger.warning(f"Failed to upload metadata: {e}")
            return False


async def reload_models() -> bool:
    """通知 Railway 重新載入模型"""
    logger.info("🔄 Requesting model reload...")

    headers = {
        "X-API-Key": RAILWAY_API_KEY
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{RAILWAY_API_URL}/api/v1/models/reload",
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"✅ Models reloaded: {result.get('message', 'OK')}")
                    logger.info(f"   Loaded models: {result.get('loaded_models', [])}")
                    return True
                else:
                    error = await response.text()
                    logger.error(f"❌ Reload failed ({response.status}): {error}")
                    return False

        except Exception as e:
            logger.error(f"❌ Reload request failed: {e}")
            return False


async def list_remote_models() -> List[Dict[str, Any]]:
    """列出 Railway 上的模型"""
    headers = {
        "X-API-Key": RAILWAY_API_KEY
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(
                f"{RAILWAY_API_URL}/api/v1/models",
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("models", [])
                else:
                    logger.error(f"❌ Failed to list remote models ({response.status})")
                    return []

        except Exception as e:
            logger.error(f"❌ Failed to list remote models: {e}")
            return []


async def delete_remote_model(filename: str) -> bool:
    """刪除 Railway 上的模型"""
    headers = {
        "X-API-Key": RAILWAY_API_KEY
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.delete(
                f"{RAILWAY_API_URL}/api/v1/models/{filename}",
                headers=headers
            ) as response:
                if response.status == 200:
                    logger.info(f"✅ Deleted: {filename}")
                    return True
                else:
                    logger.error(f"❌ Delete failed ({response.status})")
                    return False

        except Exception as e:
            logger.error(f"❌ Delete failed: {e}")
            return False


def list_local_models():
    """列出本地模型"""
    models = get_model_files()

    if not models:
        logger.info("📁 No local models found")
        return

    logger.info(f"📁 Local Models ({len(models)} files):")
    logger.info("=" * 70)

    for model in models:
        metadata = get_model_metadata(model)
        size_mb = model.stat().st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(model.stat().st_mtime)

        logger.info(f"  📦 {model.name}")
        logger.info(f"     Type: {metadata.get('model_type', 'unknown')}")
        logger.info(f"     Symbol: {metadata.get('symbol', 'unknown')}")
        logger.info(f"     Size: {size_mb:.2f} MB")
        logger.info(f"     Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("")


async def list_remote():
    """列出遠端模型"""
    models = await list_remote_models()

    if not models:
        logger.info("🌐 No remote models found")
        return

    logger.info(f"🌐 Remote Models ({len(models)} files):")
    logger.info("=" * 70)

    for model in models:
        logger.info(f"  📦 {model.get('filename', 'unknown')}")
        logger.info(f"     Type: {model.get('model_type', 'unknown')}")
        logger.info(f"     Symbol: {model.get('symbol', 'unknown')}")
        logger.info(f"     Loaded: {'✅' if model.get('loaded', False) else '❌'}")
        logger.info(f"     Uploaded: {model.get('uploaded_at', 'unknown')}")
        logger.info("")


async def sync_all(force: bool = False):
    """同步所有本地模型到 Railway"""
    models = get_model_files()

    if not models:
        logger.info("📁 No local models to sync")
        return

    logger.info(f"🔄 Syncing {len(models)} models to Railway...")

    success_count = 0
    for model in models:
        if await upload_model(model, force=force):
            success_count += 1

    logger.info(f"✅ Synced {success_count}/{len(models)} models")

    # 重新載入模型
    if success_count > 0:
        await reload_models()


def main():
    parser = argparse.ArgumentParser(
        description="模型同步腳本 - 將本地模型上傳到 Railway",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  # 列出本地模型
  python scripts/sync_models.py --list

  # 列出 Railway 上的模型
  python scripts/sync_models.py --list-remote

  # 上傳最新模型
  python scripts/sync_models.py --upload-latest

  # 上傳指定模型
  python scripts/sync_models.py --upload models/ensemble_BTCUSDT_20241212.joblib

  # 同步所有模型
  python scripts/sync_models.py --sync-all

  # 通知 Railway 重新載入模型
  python scripts/sync_models.py --reload
        """
    )

    parser.add_argument("--list", action="store_true", help="列出本地模型")
    parser.add_argument("--list-remote", action="store_true", help="列出 Railway 上的模型")
    parser.add_argument("--upload", type=str, help="上傳指定模型檔案")
    parser.add_argument("--upload-latest", action="store_true", help="上傳最新的模型")
    parser.add_argument("--upload-type", type=str, help="上傳指定類型的最新模型")
    parser.add_argument("--sync-all", action="store_true", help="同步所有本地模型")
    parser.add_argument("--reload", action="store_true", help="通知 Railway 重新載入模型")
    parser.add_argument("--delete", type=str, help="刪除 Railway 上的模型")
    parser.add_argument("--force", action="store_true", help="強制覆蓋現有模型")

    args = parser.parse_args()

    # 顯示配置
    logger.info("=" * 50)
    logger.info("🔄 Model Sync Tool")
    logger.info("=" * 50)
    logger.info(f"Railway URL: {RAILWAY_API_URL or '(未設定)'}")
    logger.info(f"Local Model Dir: {MODEL_DIR}")
    logger.info("")

    # 需要遠端連線的操作先檢查環境變數
    if any([args.list_remote, args.upload, args.upload_latest,
            args.upload_type, args.sync_all, args.reload, args.delete]):
        _require_remote_config()

    if args.list:
        list_local_models()
    elif args.list_remote:
        asyncio.run(list_remote())
    elif args.upload:
        model_path = Path(args.upload)
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        asyncio.run(upload_model(model_path, force=args.force))
    elif args.upload_latest:
        model = get_latest_model()
        if model:
            asyncio.run(upload_model(model, force=args.force))
        else:
            logger.error("❌ No models found")
    elif args.upload_type:
        model = get_latest_model(args.upload_type)
        if model:
            asyncio.run(upload_model(model, force=args.force))
        else:
            logger.error(f"❌ No {args.upload_type} models found")
    elif args.sync_all:
        asyncio.run(sync_all(force=args.force))
    elif args.reload:
        asyncio.run(reload_models())
    elif args.delete:
        asyncio.run(delete_remote_model(args.delete))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
