from serpapi import GoogleSearch
from watsonx_llm import llm_generate_gl

def search_products(query, location):
    """
    Fetch product results using SerpAPI.

    Args:
        query (str): Refined search query for the product.
        location (str): Location for the search (default: "United States").

    Returns:
        list: A list of dictionaries containing product information.
    """
    
    gl = llm_generate_gl(location)
    if not gl:
        print(f"Invalid or missing GL code for location '{location}'. Defaulting to 'US'.")
        gl = "US"  # Default fallback
    print(f"Generated country code (gl): {gl}")

    params = {
        "q": query,
        "tbm": "shop",  # Shopping mode
        "location": location,  # Add location to the search parameters
        "hl": "en",  # Language
        "gl": gl,  # Country
        "api_key": "55c9b2d43d65c2bf2769e504062abb93a6b26c9f44551786b059928eda18792a"  # Replace with your SerpAPI key
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results.get("shopping_results", [])

if __name__ == "__main__":
    # Example test query
    test_query = "Buy affordable jeans online"
    test_location = "Austin, Texas, United States"  # Specify location
    products = search_products(test_query, location=test_location)
    for product in products:
        print(f"Title: {product['title']}")
        print(f"Price: {product['price']}")
        print(f"Source: {product['source']}")
        print(f"Link: {product['link']}")
        print(f"Rating: {product.get('rating', 'N/A')}")
        print(f"Reviews: {product.get('reviews', 'N/A')}")
        print("-" * 40)