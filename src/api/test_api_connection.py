from client import NBAApiClient


def test_api():
    client = NBAApiClient()

    # Replace with an actual endpoint from your API (this is an example)
    sample_endpoint = "/scores/json/teams"
    params = {"format": "json"}

    try:
        data = client.get(sample_endpoint, params)
        print("✅ API connection successful.")
        print(f"Returned {len(data)} items.")
    except Exception as e:
        print(f"❌ API connection failed: {e}")


if __name__ == "__main__":
    test_api()
