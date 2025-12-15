# src/utils/s3_upload.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError


def upload_to_s3(
    local_path: Path,
    bucket: str,
    key: str,
    public: bool = True,
) -> Optional[str]:
    """
    Upload a local file to S3.

    Parameters
    ----------
    local_path : Path
        Local file to upload (CSV/JSON/etc.).
    bucket : str
        S3 bucket name, e.g. "oddzup-stats-2025".
    key : str
        S3 object key, e.g. "season/player_profiles_from_db_2025-26.json".
    public : bool
        If True, sets ACL to public-read so the file is reachable via HTTPS.

    Returns
    -------
    Optional[str]
        Public HTTPS URL if upload succeeds, otherwise None.
    """
    local_path = Path(local_path)

    if not local_path.exists():
        print(f"❌ Local file does not exist, cannot upload: {local_path}")
        return None

    s3 = boto3.client("s3")

    # Content type for nicer behavior in browsers / clients
    if local_path.suffix == ".json":
        content_type = "application/json"
    elif local_path.suffix == ".csv":
        content_type = "text/csv"
    else:
        content_type = "application/octet-stream"

    extra_args = {"ContentType": content_type}
    if public:
        extra_args["ACL"] = "public-read"

    try:
        s3.upload_file(
            Filename=str(local_path),
            Bucket=bucket,
            Key=key,
            ExtraArgs=extra_args,
        )
    except (BotoCoreError, ClientError) as e:
        print(f"❌ Failed to upload {local_path} to s3://{bucket}/{key}: {e}")
        return None

    # Build a nice HTTPS URL
    region = s3.meta.region_name or "us-east-1"
    if region == "us-east-1":
        url = f"https://{bucket}.s3.amazonaws.com/{key}"
    else:
        url = f"https://{bucket}.s3.{region}.amazonaws.com/{key}"

    print(f"✅ Uploaded {local_path} to s3://{bucket}/{key}")
    print(f"🌐 Public URL: {url}")
    return url
