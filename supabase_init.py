import os
from supabase import create_client, Client
from dotenv import load_dotenv
load_dotenv()

def get_client():
    url: str = os.environ.get("DB_URL")
    key: str = os.environ.get("DB_KEY")
    supabase: Client = create_client(url, key)

    return supabase

