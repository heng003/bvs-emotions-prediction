import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup Supabase
# Ensure SUPABASE_URL and SUPABASE_KEY are in your .env file
url: str = os.environ.get("SUPABASE_URL", "")
key: str = os.environ.get("SUPABASE_KEY", "")

if not url or not key:
    raise ValueError("Supabase credentials missing. Please set SUPABASE_URL and SUPABASE_KEY in .env")

supabase_client: Client = create_client(url, key)