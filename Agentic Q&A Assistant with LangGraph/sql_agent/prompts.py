from sql_agent.db import get_schema

LOV_CONTEXT = """
Example values:

customers.country → Germany, USA, France, Brazil, UK, Canada, Spain
customers.city → Berlin, London, Paris, São Paulo, Madrid, Seattle
customers.company_name → Alfreds Futterkiste, Ana Trujillo Emparedados y helados, Around the Horn
customers.contact_title → Owner, Sales Representative, Marketing Manager

employees.last_name → Davolio, Fuller, Leverling, Peacock, Buchanan, Suyama
employees.title → Sales Representative, Vice President Sales, Sales Manager
employees.country → USA, UK

categories.category_name → Beverages, Condiments, Confections, Dairy Products, Grains/Cereals, Meat/Poultry, Produce, Seafood

products.product_name → Chai, Chang, Ikura, Tofu, Pavlova, Chef Anton's Cajun Seasoning
products.discontinued → 0, 1

orders.ship_country → Germany, USA, France, Brazil, UK, Canada
orders.ship_city → Berlin, London, Paris, São Paulo, Madrid
orders.ship_via → 1, 2, 3

shippers.company_name → Speedy Express, United Package, Federal Shipping

suppliers.country → USA, UK, Germany, Japan, Sweden
suppliers.company_name → Exotic Liquids, New Orleans Cajun Delights, Tokyo Traders

territories.territory_id → 01581, 01730, 01833

us_states.state_abbr → CA, WA, NY, TX, FL
region.region_id → 1, 2, 3, 4
"""

def build_prompt(question: str) -> str:
    schema = get_schema()

    return f"""
You are an expert PostgreSQL analyst.

Use ONLY the schema below.
Do NOT invent tables or columns.

Schema:
{schema}

Helpful examples:
{LOV_CONTEXT}

Question:
{question}

Rules:
- Use valid PostgreSQL SQL
- Use explicit JOINs
- Prefer correct business logic
- Return ONLY SQL, no explanation
"""
