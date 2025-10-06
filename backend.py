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
    if not SHOP_URL or not ACCESS_TOKEN:
        raise HTTPException(500, "SHOP_URL or ACCESS_TOKEN not configured")
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
    import re

    q_raw = (body.query or "").strip()
    limit = max(1, min(20, body.limit or 5))

    def normalize(s: str) -> str:
        s = s.strip()
        s = s.replace("“", "").replace("”", "").replace("‘", "").replace("’", "")
        s = s.replace('"', "").replace("'", "")
        s = re.sub(r"[^0-9A-Za-zÀ-ÿ\s\-]", " ", s)  # 仅字母数字/空格/连字符（含重音）
        s = re.sub(r"\s+", " ", s)
        return s.strip()

# 候选查询词（原文/清洗/前2-3词 + 尾部2/3词 + 去问法前缀）
candidates: List[str] = []
if q_raw:
    candidates.append(q_raw)

    norm = normalize(q_raw)              # 例："quanto costa coordinato anahi"
    candidates.append(norm)

    parts = norm.split()

    # 前 2/3 词（原逻辑）
    if len(parts) >= 2:
        candidates.append(" ".join(parts[:2]))
    if len(parts) >= 3:
        candidates.append(" ".join(parts[:3]))

    # ✅ 尾部 2/3 词（新增）——常见真实商品名在句尾
    if len(parts) >= 2:
        candidates.append(" ".join(parts[-2:]))   # 如 "coordinato anahi"
    if len(parts) >= 3:
        candidates.append(" ".join(parts[-3:]))

    # ✅ 去掉常见问法前缀后再取尾部 2/3 词（新增）
    import re
    prefix_re = re.compile(
        r"^(quanto\s+costa|prezzo|quanto\s+viene|costa|price|prezzi|il\s+prezzo\s+di)\b",
        flags=re.IGNORECASE
    )
    tail = prefix_re.sub("", norm).strip()        # "coordinato anahi"
    if tail:
        candidates.append(tail)
        tparts = tail.split()
        if len(tparts) >= 2:
            candidates.append(" ".join(tparts[-2:]))
        if len(tparts) >= 3:
            candidates.append(" ".join(tparts[-3:]))

# 去重
cand_list: List[str] = []
seen = set()
for c in candidates:
    c2 = c.strip().lower()
    if c2 and c2 not in seen:
        seen.add(c2)
        
    items: List[Dict[str, Any]] = []
    seen_variants: set[int] = set()

    def push_item(p: Dict[str, Any], v: Dict[str, Any]):
        try:
            key = int(v["id"])
        except Exception:
            return
        if key in seen_variants:
            return
        seen_variants.add(key)
        items.append({
            "product_id": p.get("id"),
            "variant_id": v.get("id"),
            "title": p.get("title"),
            "sku": v.get("sku"),
            "price": float(v["price"]) if v.get("price") is not None else None,
            "compare_at_price": float(v["compare_at_price"]) if v.get("compare_at_price") else None,
            "available": v.get("available", True),
            "inventory_qty": v.get("inventory_quantity"),
            "currency": "EUR",
            "url": f"{PUBLIC_STORE_URL}/products/{p.get('handle','')}" if PUBLIC_STORE_URL else None
        })

    # 0) 变体ID直查
    if q_raw.isdigit():
        try:
            v = shopify_request(f"/variants/{q_raw}.json").get("variant")
            if v:
                p = shopify_request(f"/products/{v['product_id']}.json").get("product", {})
                if p:
                    push_item(p, v)
        except Exception:
            pass
        if len(items) >= limit:
            return {"items": items[:limit]}

    # 1) 精确 SKU（对每个候选词都试）
    for q in cand_list:
        try:
            vjson = shopify_request("/variants.json", params={"sku": q, "limit": 50})
            for v in vjson.get("variants", []):
                p = shopify_request(f"/products/{v['product_id']}.json").get("product", {})
                if p:
                    push_item(p, v)
                    if len(items) >= limit:
                        return {"items": items[:limit]}
        except Exception:
            pass

    # 2) 标题精确（Shopify 按 title 过滤）
    for q in cand_list:
        try:
            by_title = shopify_request("/products.json", params={"title": q, "limit": 50}).get("products", [])
            for p in by_title:
                v = (p.get("variants") or [{}])[0]
                if v and v.get("id"):
                    push_item(p, v)
                    if len(items) >= limit:
                        return {"items": items[:limit]}
        except Exception:
            pass

    # 3) 兜底包含（title 或 sku 包含任一候选）
    try:
        products = shopify_request("/products.json", params={"limit": 250}).get("products", [])
        for p in products:
            title_l = (p.get("title") or "").lower()
            hit = any(q in title_l for q in cand_list if q)
            chosen = None
            if not hit:
                for v in p.get("variants", []):
                    sku_l = (v.get("sku") or "").lower()
                    if any(q in sku_l for q in cand_list if q):
                        chosen = v
                        hit = True
                        break
            if hit:
                v = chosen or (p.get("variants") or [{}])[0]
                if v and v.get("id"):
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
- Se l'utente chiede “quanto costa / prezzo / disponibilità / SKU / variant id”, chiama SEMPRE lookup_price con query PULITA e rispondi con titolo+prezzo+disponibilità+link (max 3).
- Niente saluti quando è una domanda di prezzo: vai subito al risultato.
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
    return {"ok": True, "hint": "user message included; no system role in thread", "ts": time.time()}


@app.post("/assistant/chat")
def assistant_chat(body: ChatIn):
    try:
        if not ASSISTANT_ID:
            raise HTTPException(500, "ASSISTANT_ID not configured")

        client = get_openai_client()

        # 1) create thread
        thread = client.beta.threads.create()

        # 2) >>>>>>>>>>> 关键：把用户消息写入线程 <<<<<<<<<<
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=body.message or ""
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
