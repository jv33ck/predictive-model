# src/utils/s3_upload.py
"""
Simple S3 upload helper for CSV / JSON artifacts.

NOTE: The target bucket (`oddzup-stats-2025`) has ACLs disabled
      (Bucket Owner Enforced). That means we must NOT set any ACL
      such as "public-read" on uploads, or S3 will return
      AccessControlListNotSupported.

Public / client access should be handled via bucket policy, not ACLs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError


def _infer_content_type(path: Path) -> str:
    """Best-effort Content-Type based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "text/csv"
    if suffix == ".json":
        return "application/json"
    # Fallback generic type
    return "application/octet-stream"


def upload_to_s3(
    path: str | Path,
    bucket: str,
    key: str,
    *,
    # kept for backwards compatibility; we NO LONGER set ACLs at all
    public: Optional[bool] = None,
    content_type: Optional[str] = None,
    cache_control: Optional[str] = None,
) -> str:
    """
    Upload a local file to S3.

    Parameters
    ----------
    path : str or Path
        Local path to the file.
    bucket : str
        Target S3 bucket name.
    key : str
        S3 object key (path within bucket).
    public : bool, optional
        Ignored. Kept only for backwards compatibility. The bucket
        has ACLs disabled, so all access control must be managed
        via bucket policies instead of object ACLs.
    content_type : str, optional
        Optional explicit Content-Type. If omitted, inferred
        from file extension.
    cache_control : str, optional
        Optional Cache-Control header.

    Returns
    -------
    str
        The s3:// URL for the uploaded object.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    if content_type is None:
        content_type = _infer_content_type(file_path)

    extra_args: dict[str, str] = {}
    if content_type:
        extra_args["ContentType"] = content_type
    if cache_control:
        extra_args["CacheControl"] = cache_control

    s3 = boto3.client("s3")

    print(f"☁️ Uploading {file_path} to s3://{bucket}/{key} ...")

    try:
        if extra_args:
            s3.upload_file(str(file_path), bucket, key, ExtraArgs=extra_args)
        else:
            s3.upload_file(str(file_path), bucket, key)
    except (BotoCoreError, ClientError) as exc:
        # Surface a clear, single error
        raise RuntimeError(
            f"Failed to upload {file_path} to s3://{bucket}/{key}: {exc}"
        ) from exc

    print(f"✅ Upload complete: s3://{bucket}/{key}")
    return f"s3://{bucket}/{key}"
