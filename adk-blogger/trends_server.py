#!/usr/bin/env python3
"""
Minimal MCP server exposing a single tool: `trends`

Fast & demo-friendly:
- quick=True (default) returns only related queries (fastest)
- quick=False returns related + interest_over_time (slower)
- Tight pytrends timeouts, no retries
- Fallbacks: daily & realtime trending when related_queries is empty/errs
- No prints to stdout before handshake (stderr only for logs)
"""

import asyncio, json, os, sys
from typing import Dict, List, Optional

# MCP (low-level stdio)
from mcp import types as mcp_types
from mcp.server.lowlevel import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio

# ADK FunctionTool wrapper + schema conversion
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.mcp_tool.conversion_utils import adk_to_mcp_tool_type


def _log(msg: str):
   print(f"[trends] {msg}", file=sys.stderr)


def _pn_for_daily(geo: Optional[str]) -> str:
   """
   Map ISO-2 to pytrends daily trending `pn` names (not exhaustive).
   Defaults to united_states if unknown.
   """
   m = {
       "US": "united_states",
       "GB": "united_kingdom",
       "UK": "united_kingdom",
       "CA": "canada",
       "AU": "australia",
       "IN": "india",
       "DE": "germany",
       "FR": "france",
       "JP": "japan",
       "BR": "brazil",
   }
   return m.get((geo or "US").upper(), "united_states")


def trends(
   keyword: str,
   geo: Optional[str] = "US",              # ISO-2 ("US"), "" for worldwide
   timeframe: Optional[str] = "now 7-d",   # small window == faster
   hl: Optional[str] = "en-US",
   quick: Optional[bool] = True,           # if True => skip iot for speed
) -> Dict:
   """
   Return Google Trends signals for `keyword`.

   quick=True:
     - related.top / related.rising (fast)
     - if empty/failed: fall back to daily or realtime trending lists
   quick=False:
     - related.* + interest_over_time (slower)

   Returns JSON-safe dict (no pandas objects).
   """
   try:
       from pytrends.request import TrendReq
       import pandas as pd  # noqa: F401
   except Exception as e:
       return {"status": "error", "message": f"pytrends not installed: {e}"}

   try:
       # Allow overriding via env if you need to tweak during a demo
       connect_timeout = float(os.getenv("TRENDS_CONNECT_TIMEOUT_S", "3.05"))
       read_timeout    = float(os.getenv("TRENDS_READ_TIMEOUT_S", "8.0"))
       retries         = int(os.getenv("TRENDS_RETRIES", "0"))
       backoff         = float(os.getenv("TRENDS_BACKOFF", "0"))

       pt = TrendReq(
           hl=hl or "en-US",
           tz=360,
           retries=retries,
           backoff_factor=backoff,
           requests_args={"timeout": (connect_timeout, read_timeout)},
       )

       tf = timeframe or "now 7-d"

       # --- Build payload for keyword-dependent endpoints ---
       try:
           pt.build_payload([keyword], timeframe=tf, geo=geo or "")
       except Exception as e:
           _log(f"build_payload failed: {e}")

       # --- Related queries (fast path) ---
       related_top: List[Dict] = []
       related_rising: List[Dict] = []

       def pack(df):
           out = []
           if df is not None and not df.empty:
               for _, r in df.iterrows():
                   out.append({"query": str(r.get("query", "")), "value": int(r.get("value", 0))})
           return out

       used_fallbacks: List[str] = []

       try:
           rq = pt.related_queries() or {}
           bucket = rq.get(keyword, {}) if isinstance(rq, dict) else {}
           related_top = pack(bucket.get("top"))
           related_rising = pack(bucket.get("rising"))
       except Exception as e:
           _log(f"related_queries failed: {e}")

       # --- Fallbacks if related queries are empty ---
       if not related_top and not related_rising:
           # 1) Daily trending (requires full-name pn)
           try:
               pn_daily = _pn_for_daily(geo)
               tr = pt.trending_searches(pn=pn_daily)
               if tr is not None and not tr.empty:
                   col = tr.columns[0]
                   related_rising = [{"query": str(x), "value": 0} for x in tr[col].tolist()[:20]]
                   used_fallbacks.append(f"trending_searches:{pn_daily}")
           except Exception as e:
               _log(f"trending_searches failed: {e}")

           # 2) Realtime trending (accepts ISO-2)
           if not related_rising:
               try:
                   pn_rt = (geo or "US").upper() or "US"
                   rt = pt.realtime_trending_searches(pn=pn_rt)
                   if rt is not None and not rt.empty:
                       first_col = rt.columns[0]
                       related_rising = [{"query": str(x), "value": 0} for x in rt[first_col].tolist()[:20]]
                       used_fallbacks.append(f"realtime_trending_searches:{pn_rt}")
               except Exception as e:
                   _log(f"realtime_trending_searches failed: {e}")

           # 3) Last resort: echo the keyword so we still return ok
           if not related_top and not related_rising:
               related_top = [{"query": keyword, "value": 0}]
               used_fallbacks.append("keyword_echo")

       payload: Dict = {
           "status": "ok",
           "inputs": {
               "keyword": keyword,
               "geo": geo or "",
               "timeframe": tf,
               "hl": hl or "en-US",
               "quick": bool(quick),
           },
           "related": {
               "top": related_top,
               "rising": related_rising,
           },
       }

       # --- Optional interest_over_time (slower) ---
       if not quick:
           try:
               iot_df = pt.interest_over_time()
               iot: List[Dict] = []
               if iot_df is not None and not iot_df.empty:
                   df = iot_df.reset_index()
                   date_col = df.columns[0]  # first column is datetime after reset_index
                   # pick the first numeric column if keyword is not present
                   key_col = keyword if keyword in df.columns else next((c for c in df.columns[1:] if c != "isPartial"), None)
                   for _, row in df.iterrows():
                       iot.append({
                           "date": str(row.get(date_col, "")),
                           "value": int(row.get(key_col, 0)) if key_col else 0,
                           "isPartial": bool(row.get("isPartial", False)),
                       })
               payload["interest_over_time"] = iot
           except Exception as e:
               _log(f"interest_over_time failed: {e}")
               payload["interest_over_time"] = []

       if used_fallbacks:
           payload["fallback"] = used_fallbacks

       return payload

   except Exception as e:
       _log(f"fatal error: {e}")
       return {"status": "error", "message": f"trends failed: {e}"}


# Wrap as ADK FunctionTool
trends_tool = FunctionTool(trends)

# MCP app
app = Server("adk-trends-mcp")


@app.list_tools()
async def list_tools() -> list[mcp_types.Tool]:
   return [adk_to_mcp_tool_type(trends_tool)]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[mcp_types.Content]:
   if name != trends_tool.name:
       err = {"error": f"unknown tool '{name}'"}
       return [mcp_types.TextContent(type="text", text=json.dumps(err))]
   try:
       result = await trends_tool.run_async(args=arguments, tool_context=None)
       return [mcp_types.TextContent(type="text", text=json.dumps(result, indent=2))]
   except Exception as e:
       _log(f"call_tool error: {e}")
       err = {"error": f"Execution failed: {e}"}
       return [mcp_types.TextContent(type="text", text=json.dumps(err))]


# stdio runner
async def run_stdio():
   async with mcp.server.stdio.stdio_server() as (r, w):
       await app.run(
           r, w,
           InitializationOptions(
               server_name=app.name,
               server_version="0.1.2",
               capabilities=app.get_capabilities(
                   notification_options=NotificationOptions(),
                   experimental_capabilities={},
               ),
           ),
       )


if __name__ == "__main__":
   try:
       asyncio.run(run_stdio())
   except KeyboardInterrupt:
       pass
