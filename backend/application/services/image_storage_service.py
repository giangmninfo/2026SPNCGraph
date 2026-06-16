import uuid
import traceback
from supabase import Client
from backend.interfaces.services.image_storage_service_interface import IImageStorageService


class SupabaseImageStorageService(IImageStorageService):
    def __init__(self, supabase: Client, bucket_name: str):
        self.supabase = supabase
        self.bucket = bucket_name

    def upload(
        self,
        *,
        user_id: int,
        image_bytes: bytes,
        filename: str,
        content_type: str,
    ) -> str:
        ext = filename.split(".")[-1]
        object_key = f"users/{user_id}/{uuid.uuid4()}.{ext}"

        self.supabase.storage.from_(self.bucket).upload(
            path=object_key,
            file=image_bytes,
            file_options={
                "content-type": content_type,
                "upsert": False,
            },
        )

        return object_key
    
    def get_signed_url(self, path: str, expires_in: int = 3600) -> str:
        res = self.supabase.storage.from_(self.bucket).create_signed_url(
            path,
            expires_in,
        )
        return res["signedURL"]