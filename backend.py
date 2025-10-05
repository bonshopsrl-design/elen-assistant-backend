import os
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from dotenv import load_dotenv

load_dotenv()
SHOP_URL = os.getenv("SHOP_URL")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
PUBLIC_STORE_URL = os.getenv("PUBLIC_STORE_URL", "").rstrip("/")

app = FastAPI(title="ELEN Assistant Backend", version="1.0.0")

def shopify_request(path: str, method="GET", payload=None, params=None):
    url = f"https://{SHOP_URL}/admin/api/2025-10{path}"
    headers = {
        "X-Shopify-Access-Token": ACCESS_TOKEN,
        "Content-Type": "application/json",
    }
    r = requests.request(method, url, headers=headers, json=payload, params=params, timeout=30)
    if r.status_code >= 300:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

class PriceLookupIn(BaseModel):
    query: str
    limit: int | None = 5

class LineItem(BaseModel):
    variant_id: int
    quantity: int

class Address(BaseModel):
    first_name: str
    last_name: str | None = None
    country: str = "IT"
    province: str | None = None
    city: str
    address1: str
    address2: str | None = None
    postal_code: str

class Customer(BaseModel):
    email: str | None = None
    name: str | None = None
    phone: str | None = None

class OrderCreateIn(BaseModel):
    channel: str | None = "bot"
    currency: str | None = "EUR"
    payment_method: str
    customer: Customer | None = None
    shipping_address: Address
    lines: List[LineItem]
    notes: str | None = "ordine creato dal BOT"

@app.post("/price.lookup")
def price_lookup(body: PriceLookupIn):
    q = body.query.strip()
    limit = body.limit or 5
    products = shopify_request("/products.json", params={"limit": 50}).get("products", [])
    items = []
    for p in products:
        for v in p.get("variants", []):
            sku = (v.get("sku") or "").lower()
            if q.lower() in sku or q.lower() in p.get("title", "").lower():
                items.append({
                    "product_id": p["id"],
                    "variant_id": v["id"],
                    "title": p["title"],
                    "sku": v["sku"],
                    "price": v["price"],
                    "available": v.get("available", True),
                    "url": f"{PUBLIC_STORE_URL}/products/{p.get('handle','')}"
                })
    return {"items": items[:limit]}

@app.post("/order.create")
def order_create(body: OrderCreateIn):
    payload = {
        "order": {
            "email": body.customer.email if body.customer else None,
            "phone": body.customer.phone if body.customer else None,
            "shipping_address": body.shipping_address.model_dump(),
            "billing_address": body.shipping_address.model_dump(),
            "line_items": [li.model_dump() for li in body.lines],
            "currency": body.currency,
            "note": body.notes,
            "tags": ["bot", body.channel]
        }
    }
    if body.payment_method == "prepaid":
        payload["order"]["financial_status"] = "pending"
    data = shopify_request("/orders.json", method="POST", payload=payload)
    order = data.get("order", {})
    return {"order_id": order.get("id"), "name": order.get("name"), "total": order.get("total_price")}
