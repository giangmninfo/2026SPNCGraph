from backend.settings import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
from supabase import create_client
import os

def create_supabase_client():
    return create_client(
        SUPABASE_URL,
        SUPABASE_SERVICE_ROLE_KEY,
    )
