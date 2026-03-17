import re

ALLOWED_SCHEMA = {

    "customers": {
        "customer_id", "company_name", "contact_name", "contact_title",
        "address", "city", "region", "postal_code", "country",
        "phone", "fax"
    },

    "orders": {
        "order_id", "customer_id", "employee_id", "order_date",
        "required_date", "shipped_date", "ship_via", "freight",
        "ship_name", "ship_address", "ship_city", "ship_region",
        "ship_postal_code", "ship_country"
    },

    "order_details": {
        "order_id", "product_id", "unit_price", "quantity", "discount"
    },

    "products": {
        "product_id", "product_name", "supplier_id", "category_id",
        "quantity_per_unit", "unit_price", "units_in_stock",
        "units_on_order", "reorder_level", "discontinued"
    },

    "employees": {
        "employee_id", "last_name", "first_name", "title",
        "title_of_courtesy", "birth_date", "hire_date",
        "address", "city", "region", "postal_code",
        "country", "home_phone", "extension", "reports_to"
    },

    "categories": {
        "category_id", "category_name", "description"
    },

    "suppliers": {
        "supplier_id", "company_name", "contact_name",
        "contact_title", "address", "city",
        "region", "postal_code", "country",
        "phone", "fax", "homepage"
    },

    "shippers": {
        "shipper_id", "company_name", "phone"
    },

    "territories": {
        "territory_id", "territory_description", "region_id"
    },

    "region": {
        "region_id", "region_description"
    },

    "employee_territories": {
        "employee_id", "territory_id"
    },

    "customer_demographics": {
        "customer_type_id", "customer_desc"
    },

    "customer_customer_demo": {
        "customer_id", "customer_type_id"
    },

    "us_states": {
        "state_id", "state_name", "state_abbr", "state_region"
    }
}

FORBIDDEN_KEYWORDS = {
    "insert", "update", "delete", "drop",
    "alter", "truncate", "create"
}

SQL_OPERATORS = {
    "+", "-", "*", "/", "%", "=", "<", ">", "<=", ">=", "<>", "!="
}

SQL_FUNCTIONS = {
    "sum", "count", "avg", "min", "max", "distinct"
}


def validate_sql(sql: str) -> None:
    sql_lower = sql.lower()

    # 1. SELECT only
    if not sql_lower.strip().startswith("select"):
        raise ValueError("Only SELECT queries are allowed")

    # 2. Block destructive keywords
    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", sql_lower):
            raise ValueError(f"Forbidden SQL keyword detected: {keyword}")

    # 3. Extract tables
    table_matches = re.findall(r"\bfrom\s+(\w+)|\bjoin\s+(\w+)", sql_lower)
    tables = {t for pair in table_matches for t in pair if t}

    if not tables:
        raise ValueError("No tables found in query")

    for table in tables:
        if table not in ALLOWED_SCHEMA:
            raise ValueError(f"Unknown table referenced: {table}")

    # 4. Extract SELECT clause
    select_match = re.search(r"\bselect\s+(.*?)\bfrom\b", sql_lower, re.DOTALL)
    if not select_match:
        return  # nothing to validate

    select_part = select_match.group(1)

    aliases = set(
        re.findall(r"\bas\s+(\w+)", select_part)
    )


    tokens = re.split(r"[,\s]+", select_part)

    for token in tokens:
        token = token.strip().strip("(),")

        if not token:
            continue

        # ignore numbers 
        if re.fullmatch(r"\d+(\.\d+)?", token):
            continue

        # ignore operators
        if token in SQL_OPERATORS:
            continue

        # handle functions like count(*), avg(price), sum(qty)
        if "(" in token and ")" in token:
            fn_name = token.split("(")[0].lower()
            if fn_name in SQL_FUNCTIONS:
                continue

        # ignore SQL functions
        if token in SQL_FUNCTIONS:
            continue

        # ignore aliases
        if token == "as":
            continue

        # strip table alias
        if "." in token:
            _, token = token.split(".", 1)

        # final column validation
        if token != "*" and token not in aliases and not any(
            token in ALLOWED_SCHEMA[t] for t in tables
        ):
            raise ValueError(f"Unknown column referenced: {token}")
