# scanner.py

import requests


def fetch_product(barcode: str) -> dict:
    url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"

    try:
        res = requests.get(
            url,
            timeout=5,
            headers={
                "User-Agent": "Mozilla/5.0"  # 🔥 IMPORTANT (fixes many failures)
            }
        )
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: {e}")

    # 🔥 Better error handling
    if res.status_code != 200:
        raise Exception(f"API error: status {res.status_code}")

    data = res.json()

    # DEBUG (optional)
    # print(data)

    if data.get("status") != 1:
        raise Exception("Product not found in OpenFoodFacts")

    product = data.get("product", {})

    return {
        "name": product.get("product_name", "Unknown"),
        "quantity": product.get("quantity", None)
    }