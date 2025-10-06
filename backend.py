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

# ============== Price Lookup (paginated + smart prefixes) ==============
@app.post("/price.lookup")
def price_lookup(body: PriceLookupIn):
    import re
    from math import inf

    q_raw = (body.query or "").strip()
    limit = max(1, min(20, body.limit or 5))

    # —— 扫描配置：一次最多扫 PAGE_SIZE 条，最多扫 MAX_PAGES 页（总量 ~= PAGE_SIZE * MAX_PAGES）
    PAGE_SIZE = int(os.getenv("LOOKUP_PAGE_SIZE", "250"))     # Shopify 允许到 250
    MAX_PAGES = int(os.getenv("LOOKUP_MAX_PAGES", "8"))       # 可按店铺体量调大，如 8 => 2000 条

    # —— 意大利常见“品类/前缀”词；在标题最前面时会尝试剥离
    CATEGORY_PREFIXES = {
        "coordinato", "abito", "gonna", "pantalone", "vestito",
        "tuta", "maglia", "maglione", "top", "camicia", "camicetta",
        "giacca", "cappotto", "felpa", "jeans", "shorts", "piumino",
        "cardigan", "salopette", "body", "leggings", "scarpa",
        "sandalo", "stivale", "borsa", "cintura", "set", "completo"
    }

    # 1) 规范化（保留拉丁扩展和空格/连字符）
    def normalize(s: str) -> str:
        s = s.strip()
        s = s.replace("“", "").replace("”", "").replace("‘", "").replace("’", "")
        s = s.replace('"', "").replace("'", "")
        s = re.sub(r"[^0-9A-Za-zÀ-ÿ\s\-]", " ", s)
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    # 2) 从问句生成“很可能是品名”的候选集合
    candidates: list[str] = []
    if q_raw:
        candidates.append(q_raw)

        norm = normalize(q_raw)       # 如： "quanto costa coordinato anahi"
        candidates.append(norm)

        parts = norm.split()

        # —— 前 2 / 3 词
        if len(parts) >= 2:
            candidates.append(" ".join(parts[:2]))
        if len(parts) >= 3:
            candidates.append(" ".join(parts[:3]))

        # —— 尾部 2 / 3 词（真实商品名常在句尾）
        if len(parts) >= 2:
            candidates.append(" ".join(parts[-2:]))      # "coordinato anahi"
        if len(parts) >= 3:
            candidates.append(" ".join(parts[-3:]))

        # —— 去掉常见问法前缀后再取尾部 2 / 3 词
        prefix_re = re.compile(
            r"^(quanto\s+costa|prezzo|quanto\s+viene|costa|price|prezzi|il\s+prezzo\s+di)\b",
            flags=re.IGNORECASE
        )
        tail = prefix_re.sub("", norm).strip()           # "coordinato anahi"
        if tail:
            candidates.append(tail)
            tparts = tail.split()
            if len(tparts) >= 2:
                candidates.append(" ".join(tparts[-2:]))
            if len(tparts) >= 3:
                candidates.append(" ".join(tparts[-3:]))

        # —— 如果句子以“品类前缀 + 名称”，生成“去前缀”的候选
        def strip_category_prefix(s: str) -> str:
            ps = s.lower().split()
            if ps and ps[0] in CATEGORY_PREFIXES:
                return " ".join(ps[1:])
            return s

        for c in list(candidates):
            c2 = strip_category_prefix(c)
            if c2 and c2 != c:
                candidates.append(c2)

    # 去重小写化
    cand_list: list[str] = []
    seen = set()
    for c in candidates:
        c2 = c.strip().lower()
        if c2 and c2 not in seen:
            seen.add(c2)
            cand_list.append(c2)

    # ======== 命中与排序 ========
    items: list[Dict[str, Any]] = []
    seen_variants: set[int] = set()

    def push_item(p: Dict[str, Any], v: Dict[str, Any], score: float):
        """合并去重 + 简单打分排序（分数越小越好）"""
        try:
            key = int(v["id"])
        except Exception:
            return
        if key in seen_variants:
            return
        seen_variants.add(key)

        items.append({
            "_score": score,
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

    def score_match(title_l: str, sku_l: str) -> float:
        """
        简单评分：
        - 任何一个候选 q 完全包含在 title 中：score = 0
        - token 交集 >= 2：score = 1
        - sku 命中：score = 0.5
        - 否则：+inf
        """
        best = inf
        title_tokens = set(title_l.split())
        for q in cand_list:
            if q in title_l:
                best = min(best, 0.0)
            else:
                q_tokens = set(q.split())
                if len(q_tokens) >= 2 and len(title_tokens & q_tokens) >= 2:
                    best = min(best, 1.0)
            if q and q in sku_l:
                best = min(best, 0.5)
        return best

    # ===== 0) 变体ID直查（优先、极快）
    if q_raw.isdigit():
        try:
            v = shopify_request(f"/variants/{q_raw}.json").get("variant")
            if v:
                p = shopify_request(f"/products/{v['product_id']}.json").get("product", {})
                if p:
                    t_l = (p.get("title") or "").lower()
                    s_l = (v.get("sku") or "").lower()
                    push_item(p, v, score_match(t_l, s_l))
        except Exception:
            pass
        if len(items) >= limit:
            items.sort(key=lambda it: it["_score"])
            return {"items": [ {k:v for k,v in it.items() if k!="_score"} for it in items[:limit] ]}

    # ===== 1) SKU 精确查（对每个候选词都试）
    for q in cand_list:
        try:
            vjson = shopify_request("/variants.json", params={"sku": q, "limit": 50})
            for v in vjson.get("variants", []):
                p = shopify_request(f"/products/{v['product_id']}.json").get("product", {})
                if p:
                    t_l = (p.get("title") or "").lower()
                    s_l = (v.get("sku") or "").lower()
                    push_item(p, v, score_match(t_l, s_l))
                    if len(items) >= limit:
                        items.sort(key=lambda it: it["_score"])
                        return {"items": [ {k:v for k,v in it.items() if k!="_score"} for it in items[:limit] ]}
        except Exception:
            pass

    # ===== 2) 标题精确参数（Shopify 的 title= 并非全文检索，但先试试）
    for q in cand_list:
        try:
            by_title = shopify_request("/products.json", params={"title": q, "limit": 50}).get("products", [])
            for p in by_title:
                v = (p.get("variants") or [{}])[0]
                if v and v.get("id"):
                    t_l = (p.get("title") or "").lower()
                    s_l = (v.get("sku") or "").lower()
                    push_item(p, v, score_match(t_l, s_l))
                    if len(items) >= limit:
                        items.sort(key=lambda it: it["_score"])
                        return {"items": [ {k:v for k,v in it.items() if k!="_score"} for it in items[:limit] ]}
        except Exception:
            pass

    # ===== 3) 大量商品的“分页兜底包含匹配” =====
    since_id = 0
    pages = 0
    while pages < MAX_PAGES:
        try:
            page = shopify_request("/products.json", params={
                "limit": PAGE_SIZE,
                "since_id": since_id,     # 递增 id 分页
            }).get("products", [])
        except Exception:
            break

        if not page:
            break

        for p in page:
            v_list = p.get("variants") or []
            if not v_list:
                continue
            # 用第一个变体代表这个商品（或也可都压入，但会占用配额）
            v = v_list[0]
            t_l = (p.get("title") or "").lower()
            s_l = (v.get("sku") or "").lower()

            sc = score_match(t_l, s_l)
            if sc < inf:
                push_item(p, v, sc)
                # 小优化：有足够高质量命中可提前结束
                if len(items) >= limit and sc <= 0.5:
                    break

            since_id = max(since_id, int(p.get("id", since_id)))

        pages += 1
        # 若这一页数量不足 PAGE_SIZE，说明已无更多
        if len(page) < PAGE_SIZE:
            break
        # 小优化：有足够结果且都很优，就可以收手
        if len(items) >= limit and min(it["_score"] for it in items) <= 0.5:
            break

    # 排序 + 截断 + 去掉内部字段
    items.sort(key=lambda it: it["_score"])
    return {"items": [ {k:v for k,v in it.items() if k!="_score"} for it in items[:limit] ]}

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
