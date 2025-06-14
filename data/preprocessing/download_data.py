import os
import finlab
from finlab import data
from dotenv import load_dotenv
load_dotenv()

finlab_api_key = os.getenv("FINLAB_API_KEY")

finlab.login(finlab_api_key)

