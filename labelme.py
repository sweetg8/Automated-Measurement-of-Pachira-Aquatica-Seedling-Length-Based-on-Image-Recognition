# Define the URL where Label Studio is accessible and the API key for your user account
LABEL_STUDIO_URL = 'http://localhost:8080'
# API key is available at the Account & Settings > Access Tokens page in Label Studio UI
API_KEY = 'd6f8a2622d39e9d89ff0dfef1a80ad877f4ee9e3'

# Import the SDK and the client module
from label_studio_sdk.client import LabelStudio

# Connect to the Label Studio API and check the connection
ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)
