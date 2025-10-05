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
# ====== 追加：Assistant bridge（中间层） ======
import json, time
from typing import Optional
from pydantic import BaseModel

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASSISTANT_ID = os.getenv("ASSISTANT_ID")

# 如果你装的是 openai 官方 SDK 1.x：
from openai import OpenAI
oa_client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """
Sei l'assistente clienti di ELEN MODA. Regole:
- Prezzi e disponibilità vanno sempre presi dal backend (funzione lookup_price).
- Non creare ordini senza conferma esplicita del cliente.
- Per creare ordini servono: nome, telefono, email (se disponibile), indirizzo completo, metodo di pagamento (prepagato/COD), e la lista prodotti (variant_id/sku + quantità).
- COD solo in Italia, limite importo €150 (se oltre, proponi prepagato).
- Ricapitola e chiedi conferma 'Sì/No' prima di creare l'ordine.
- Dopo ordine: mostra numero ordine e tempistiche; tracking arriverà via email dopo la spedizione.
- Se l'utente chiede solo “quanto costa…”, usa lookup_price e rispondi con prezzo, disponibilità, e link prodotto.
"""

class ChatIn(BaseModel):
    message: str
    user_id: Optional[str] = None

def call_internal_tool(name: str, args: dict):
    """
    将 Assistants 的函数调用映射到你后端已有的两个接口。
    因为就在同一进程里，我们直接调内部函数以减少网络耗时；
    你也可以用 requests.post 调用 'http://127.0.0.1:10000/price.lookup' 等。
    """
    if name == "lookup_price":
        return price_lookup(PriceLookupIn(**args))
    elif name == "create_order":
        return order_create(OrderCreateIn(**args))
    else:
        return {"error": f"unknown tool {name}"}

@app.post("/assistant/chat")
def assistant_chat(body: ChatIn):
    """
    输入: { "message": "Quanto costa 5PACK701?", "user_id": "whatsapp:+39..." }
    输出: { "reply": "..." }
    """
    if not OPENAI_API_KEY or not ASSISTANT_ID:
        raise HTTPException(500, "OPENAI_API_KEY or ASSISTANT_ID not configured")

    # 1) 新建对话线程
    thread = oa_client.beta.threads.create()

    # 2) 加系统指令（可选：也可以放到 Assistant 的“Instructions”里）
    oa_client.beta.threads.messages.create(
        thread_id=thread.id,
        role="system",
        content=SYSTEM_PROMPT
    )

    # 3) 塞入用户消息
    oa_client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=body.message
    )

    # 4) 运行 assistant
    run = oa_client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=ASSISTANT_ID
    )

    # 5) 处理函数调用循环
    while True:
        run = oa_client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        status = run.status
        if status == "requires_action":
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            outputs = []
            for tc in tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                result = call_internal_tool(name, args)
                outputs.append({"tool_call_id": tc.id, "output": json.dumps(result)})
            oa_client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread.id,
                run_id=run.id,
                tool_outputs=outputs
            )
        elif status in ("queued", "in_progress"):
            time.sleep(0.6)
        else:
            break

    # 6) 取最终回复文本
    msgs = oa_client.beta.threads.messages.list(thread_id=thread.id)
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
