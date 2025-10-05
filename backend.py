import os
import json
import time
import traceback
from typing import Dict, Any, List, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ============== ENV ==============
load_dotenv()
SHOP_URL = os.getenv("SHOP_URL")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
PUBLIC_STORE_URL = (os.getenv("PUBLIC_STORE_URL") or "").rstrip("/")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASSISTANT_ID = os.getenv("ASSISTANT_ID")
OPENAI_PROJECT = os.getenv("OPENAI_PROJECT")  # optional

app = FastAPI(title="ELEN Assistant Backend", version="1.0.0")

# ============== Shopify helper ==============
def shopify_request(path: str, method: str = "GET", payload=None, params=None):
    url = f"https://{SHOP_URL}/admin/api/2025-10{path}"
    headers = {
        "X-Shopify-Access-Token": ACCESS_TOKEN,
        "Content-Type": "application/json",
    }
    r = requests.request(method, url, headers=headers, json=payload, params=params, timeout=30)
    if r.status_code >= 300:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

# ============== Models ==============
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

# ============== Price Lookup (enhanced) ==============
@app.post("/price.lookup")
def price_lookup(body: PriceLookupIn):
    q = (body.query or "").strip()
    limit = body.limit or 5
    items: List[Dict[str, Any]] = []
    seen: set[int] = set()

    def push_item(p: Dict[str, Any], v: Dict[str, Any]):
        key = int(v["id"])
        if key in seen:
            return
        seen.add(key)
        items.append({
            "product_id": p.get("id"),
            "variant_id": v["id"],
            "title": p.get("title"),
            "sku": v.get("sku"),
            "price": float(v["price"]),
            "compare_at_price": float(v["compare_at_price"]) if v.get("compare_at_price") else None,
            "available": v.get("available", True),
            "inventory_qty": v.get("inventory_quantity"),
            "currency": "EUR",
            "url": f"{PUBLIC_STORE_URL}/products/{p.get('handle','')}" if PUBLIC_STORE_URL else None
        })

    # 0) variant_id lookup
    if q.isdigit():
        try:
            v = shopify_request(f"/variants/{q}.json").get("variant")
            if v:
                p = shopify_request(f"/products/{v['product_id']}.json").get("product", {})
                push_item(p, v)
        except Exception:
            pass
        if len(items) >= limit:
            return {"items": items[:limit]}

    # 1) exact SKU
    try:
        vjson = shopify_request("/variants.json", params={"sku": q, "limit": 50})
        for v in vjson.get("variants", []):
            p = shopify_request(f"/products/{v['product_id']}.json").get("product", {})
            push_item(p, v)
            if len(items) >= limit:
                return {"items": items[:limit]}
    except Exception:
        pass

    # 2) title fuzzy
    if q:
        try:
            by_title = shopify_request("/products.json", params={"title": q, "limit": 50}).get("products", [])
            for p in by_title:
                v = p["variants"][0]
                push_item(p, v)
                if len(items) >= limit:
                    return {"items": items[:limit]}
        except Exception:
            pass

    # 3) fallback contains (title/sku)
    try:
        products = shopify_request("/products.json", params={"limit": 250}).get("products", [])
        qlow = q.lower()
        for p in products:
            title = (p.get("title") or "").lower()
            hit = (qlow in title) if q else False
            chosen = None
            for v in p.get("variants", []):
                sku = (v.get("sku") or "").lower()
                if q and (qlow in sku):
                    chosen = v
                    hit = True
                    break
            if hit:
                v = chosen or p["variants"][0]
                push_item(p, v)
                if len(items) >= limit:
                    break
    except Exception:
        pass

    return {"items": items[:limit]}

# ============== Order Create ==============
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
    return {
        "order_id": order.get("id"),
        "name": order.get("name"),
        "total": order.get("total_price"),
        "currency": order.get("currency")
    }

# ============== Peek products (debug) ==============
@app.get("/products.peek")
def products_peek(limit: int = 30):
    js = shopify_request("/products.json", params={"limit": min(250, max(1, limit))})
    out = []
    for p in js.get("products", []):
        first = (p.get("variants") or [{}])[0]
        out.append({
            "id": p.get("id"),
            "title": p.get("title"),
            "handle": p.get("handle"),
            "first_variant_id": first.get("id"),
            "first_variant_sku": first.get("sku"),
            "price": first.get("price"),
        })
    return {"products": out}

# ============== Assistant Bridge ==============
SYSTEM_PROMPT = """
Sei l'assistente clienti di ELEN MODA. Regole:
- Prezzi e disponibilità vanno sempre presi dal backend (funzione lookup_price).
- Non creare ordini senza conferma esplicita del cliente.
- Per creare ordini servono: nome, telefono, email (se disponibile), indirizzo completo, metodo di pagamento (prepagato/COD), e la lista prodotti (variant_id/sku + quantità).
- COD solo in Italia, limite importo €150 (se oltre, proponi prepagato).
- Ricapitola e chiedi conferma 'Sì/No' prima di creare l'ordine.
- Dopo ordine: mostra numero ordine e tempistiche; il tracking arriva via email dopo la spedizione.
- Se l'utente chiede solo “quanto costa…”, usa lookup_price e rispondi con prezzo, disponibilità, e link prodotto.
- Se ci sono più risultati, elenca i migliori (max 3) con puntini elenco.
"""

class ChatIn(BaseModel):
    message: str
    user_id: Optional[str] = None

def call_internal_tool(name: str, args: dict):
    if name == "lookup_price":
        return price_lookup(PriceLookupIn(**args))
    if name == "create_order":
        return order_create(OrderCreateIn(**args))
    return {"error": f"unknown tool {name}"}

from openai import OpenAI
def get_openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise HTTPException(500, "OPENAI_API_KEY not configured")
    return OpenAI(api_key=OPENAI_API_KEY, project=OPENAI_PROJECT) if OPENAI_PROJECT else OpenAI(api_key=OPENAI_API_KEY)

@app.get("/__version")
def version():
    return {"ok": True, "hint": "no system role", "ts": time.time()}

@app.post("/assistant/chat")
def assistant_chat(body: ChatIn):
    try:
        if not ASSISTANT_ID:
            raise HTTPException(500, "ASSISTANT_ID not configured")

        client = get_openai_client()

        # 1) create thread
        thread = client.beta.threads.create()

        # 2) ONLY user message (no system role!)
        client.beta.threads.messages.create(
            thread_id=thread.id, role="user", content=body.message
        )

        # 3) run with instructions (= system prompt)
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=ASSISTANT_ID,
            instructions=SYSTEM_PROMPT
        )

        # 4) tool loop
        while True:
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if run.status == "requires_action":
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                outputs = []
                for tc in tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments or "{}")
                    result = call_internal_tool(name, args)
                    outputs.append({"tool_call_id": tc.id, "output": json.dumps(result)})
                client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread.id, run_id=run.id, tool_outputs=outputs
                )
            elif run.status in ("queued", "in_progress"):
                time.sleep(0.6)
            else:
                break

        # 5) final reply
        msgs = client.beta.threads.messages.list(thread_id=thread.id)
        reply_text = ""
        for m in reversed(msgs.data):
            if m.role == "assistant":
                for c in m.content:
                    if c.type == "text":
                        reply_text = c.text.value
                        break
                if reply_text:
                    break

        return {"reply": reply_text or "Nessuna risposta."}

    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    except Exception as e:
        print("ASSISTANT_CHAT_ERROR:", repr(e))
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})
