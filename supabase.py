
import os
from supabase import create_client, Client

def get_client():
    url= os.environ.get("SUPABASE_URL")
    key= os.environ.get("SUPABASE_KEY")
    supabase = create_client(url, key)
    return supabase
