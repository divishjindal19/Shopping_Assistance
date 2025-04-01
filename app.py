from flask import Flask, render_template, request
from watsonx_llm import refine_query, generate_comparison_table
from serp_api import search_products

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    """
    Render the homepage with the form.
    """
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    """
    Process the search query submitted by the user.
    """
    # Get user input
    user_input = request.form["query"]
    location = request.form.get("location", "United States")

    # Refine user query
    refined_query_response = refine_query(user_input,location)
    # refined_query = refined_query_response["refined_query"]
    refined_query = f"{refined_query_response['refined_query']} {refined_query_response['additional_info']}".strip()


    # Fetch products using SerpAPI
    products = search_products(refined_query, location=location)
    if not products:
        return render_template("index.html", error="No products found for your query.")

    # Generate comparison table and summary
    comparison_table, summary = generate_comparison_table(products)

    # Render results on the same page
    return render_template(
        "index.html",
        refined_query=refined_query,
        comparison_table=comparison_table,
        summary=summary
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)