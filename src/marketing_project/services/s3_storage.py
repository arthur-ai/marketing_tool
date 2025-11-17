"""
S3 storage utility for Marketing Project.

This module provides utilities for uploading and downloading files to/from AWS S3.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("marketing_project.services.s3_storage")

# Try to import boto3
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not available. S3 functionality will be disabled.")


class S3Storage:
    """Utility class for S3 operations."""

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        region: Optional[str] = None,
        prefix: str = "content/",
    ):
        """
        Initialize S3 storage client.

        Args:
            bucket_name: S3 bucket name (defaults to AWS_S3_BUCKET env var)
            region: AWS region (defaults to AWS_S3_REGION or AWS_REGION env var)
            prefix: S3 key prefix for all operations (default: "content/")
        """
        self.bucket_name = bucket_name or os.getenv("AWS_S3_BUCKET")
        self.region = region or os.getenv("AWS_S3_REGION") or os.getenv("AWS_REGION")
        self.prefix = prefix.rstrip("/") + "/" if prefix else ""

        if not BOTO3_AVAILABLE:
            logger.error("boto3 is not available. S3 operations will fail.")
            self.s3_client = None
            return

        if not self.bucket_name:
            logger.warning("S3 bucket name not configured. S3 operations will fail.")
            self.s3_client = None
            return

        try:
            # Initialize S3 client
            # boto3 will use default credentials from environment, IAM role, or credentials file
            self.s3_client = boto3.client("s3", region_name=self.region)
            logger.info(
                f"Initialized S3 client for bucket '{self.bucket_name}' in region '{self.region}'"
            )
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            self.s3_client = None

    def is_available(self) -> bool:
        """Check if S3 is available and configured."""
        return (
            BOTO3_AVAILABLE
            and self.s3_client is not None
            and self.bucket_name is not None
        )

    def _get_s3_key(self, file_path: str) -> str:
        """Convert local file path to S3 key."""
        # Remove leading slash if present
        file_path = file_path.lstrip("/")
        # Combine prefix with file path
        return f"{self.prefix}{file_path}"

    async def upload_file(
        self, local_path: str, s3_key: Optional[str] = None
    ) -> Optional[str]:
        """
        Upload a file to S3.

        Args:
            local_path: Local file path to upload
            s3_key: S3 key (object name). If not provided, uses the file name with prefix.

        Returns:
            S3 key if successful, None otherwise
        """
        if not self.is_available():
            logger.warning("S3 is not available. Cannot upload file.")
            return None

        try:
            if s3_key is None:
                # Use the filename with prefix
                filename = Path(local_path).name
                s3_key = self._get_s3_key(filename)
            else:
                # Remove leading slash if present
                s3_key = s3_key.lstrip("/")
                # Apply prefix only if prefix is configured and not already present
                if self.prefix and not s3_key.startswith(self.prefix):
                    s3_key = self._get_s3_key(s3_key)

            logger.info(f"Uploading {local_path} to s3://{self.bucket_name}/{s3_key}")

            # Upload file
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)

            logger.info(f"Successfully uploaded to s3://{self.bucket_name}/{s3_key}")
            return s3_key

        except FileNotFoundError:
            logger.error(f"File not found: {local_path}")
            return None
        except ClientError as e:
            logger.error(f"AWS S3 error uploading {local_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error uploading {local_path}: {e}")
            return None

    async def upload_fileobj(
        self, file_obj: Any, s3_key: str, content_type: Optional[str] = None
    ) -> bool:
        """
        Upload a file-like object to S3.

        Args:
            file_obj: File-like object to upload
            s3_key: S3 key (object name)
            content_type: Content type (MIME type) of the file

        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logger.warning("S3 is not available. Cannot upload file.")
            return False

        try:
            s3_key = self._get_s3_key(s3_key)
            extra_args = {}
            if content_type:
                extra_args["ContentType"] = content_type

            logger.info(f"Uploading file object to s3://{self.bucket_name}/{s3_key}")

            self.s3_client.upload_fileobj(
                file_obj, self.bucket_name, s3_key, ExtraArgs=extra_args
            )

            logger.info(f"Successfully uploaded to s3://{self.bucket_name}/{s3_key}")
            return True

        except ClientError as e:
            logger.error(f"AWS S3 error uploading file object: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error uploading file object: {e}")
            return False

    async def upload_json(self, data: Dict[str, Any], s3_key: str) -> Optional[str]:
        """
        Upload JSON data to S3.

        Args:
            data: Dictionary to upload as JSON
            s3_key: S3 key (object name)

        Returns:
            S3 key if successful, None otherwise
        """
        if not self.is_available():
            logger.warning("S3 is not available. Cannot upload JSON.")
            return None

        try:
            s3_key = self._get_s3_key(s3_key)
            json_str = json.dumps(data, indent=2, ensure_ascii=False)

            logger.info(f"Uploading JSON to s3://{self.bucket_name}/{s3_key}")

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json_str.encode("utf-8"),
                ContentType="application/json",
            )

            logger.info(
                f"Successfully uploaded JSON to s3://{self.bucket_name}/{s3_key}"
            )
            return s3_key

        except ClientError as e:
            logger.error(f"AWS S3 error uploading JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error uploading JSON: {e}")
            return None

    async def download_file(self, s3_key: str, local_path: str) -> bool:
        """
        Download a file from S3.

        Args:
            s3_key: S3 key (object name)
            local_path: Local file path to save to

        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logger.warning("S3 is not available. Cannot download file.")
            return False

        try:
            s3_key = self._get_s3_key(s3_key)
            logger.info(f"Downloading s3://{self.bucket_name}/{s3_key} to {local_path}")

            # Ensure directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            self.s3_client.download_file(self.bucket_name, s3_key, local_path)

            logger.info(f"Successfully downloaded to {local_path}")
            return True

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.warning(f"File not found in S3: {s3_key}")
            else:
                logger.error(f"AWS S3 error downloading {s3_key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error downloading {s3_key}: {e}")
            return False

    async def list_files(
        self, prefix: Optional[str] = None, max_keys: int = 1000
    ) -> List[str]:
        """
        List files in S3 bucket.

        Args:
            prefix: S3 key prefix to filter by (defaults to instance prefix)
            max_keys: Maximum number of keys to return

        Returns:
            List of S3 keys
        """
        if not self.is_available():
            logger.warning("S3 is not available. Cannot list files.")
            return []

        try:
            search_prefix = prefix or self.prefix
            logger.info(
                f"Listing files in s3://{self.bucket_name} with prefix '{search_prefix}'"
            )

            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=search_prefix, MaxKeys=max_keys
            )

            if "Contents" not in response:
                return []

            keys = [obj["Key"] for obj in response["Contents"]]
            logger.info(f"Found {len(keys)} files in S3")
            return keys

        except ClientError as e:
            logger.error(f"AWS S3 error listing files: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error listing files: {e}")
            return []

    async def get_file_content(self, s3_key: str) -> Optional[bytes]:
        """
        Get file content from S3 as bytes.

        Args:
            s3_key: S3 key (object name)

        Returns:
            File content as bytes, or None if error
        """
        if not self.is_available():
            logger.warning("S3 is not available. Cannot get file content.")
            return None

        try:
            s3_key = self._get_s3_key(s3_key)
            logger.info(f"Getting file content from s3://{self.bucket_name}/{s3_key}")

            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            content = response["Body"].read()

            logger.info(f"Successfully retrieved {len(content)} bytes from S3")
            return content

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.warning(f"File not found in S3: {s3_key}")
            else:
                logger.error(f"AWS S3 error getting file content: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting file content: {e}")
            return None

    async def delete_file(self, s3_key: str) -> bool:
        """
        Delete a file from S3.

        Args:
            s3_key: S3 key (object name)

        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logger.warning("S3 is not available. Cannot delete file.")
            return False

        try:
            s3_key = self._get_s3_key(s3_key)
            logger.info(f"Deleting s3://{self.bucket_name}/{s3_key}")

            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)

            logger.info(f"Successfully deleted s3://{self.bucket_name}/{s3_key}")
            return True

        except ClientError as e:
            logger.error(f"AWS S3 error deleting {s3_key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting {s3_key}: {e}")
            return False

    async def file_exists(self, s3_key: str) -> bool:
        """
        Check if a file exists in S3.

        Args:
            s3_key: S3 key (object name)

        Returns:
            True if file exists, False otherwise
        """
        if not self.is_available():
            return False

        try:
            s3_key = self._get_s3_key(s3_key)
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            logger.error(f"Error checking file existence: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking file existence: {e}")
            return False
