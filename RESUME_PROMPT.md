# 项目重启 Prompt

## 一句话背景
我们正在开发一个**零售全渠道客户行为分析平台**，核心代码位于 `/Users/poetryboat/cc/Customer/`，已完成产品模块整合，下一步开发 Chatbot 模块。

---

## 重启 Prompt（直接复制使用）

```
I'm building a customer behavior analytics system. Project path: /Users/poetryboat/cc/Customer/

Please read PROJECT_DOCUMENT.md first to understand the full project, then we continue development.
The project has three modules:
1. Product module (✅ Done): integrated into pyt/customer_analytics_dashboard_enhanced.py — sales/rating rankings, VADER sentiment analysis, per-category word clouds, all interactive Plotly charts
2. Customer module (✅ Done): pyt/customer_analytics_dashboard_enhanced.py — clustering personas, RFM, lifecycle, conversion funnel, retention matrix, interactive HTML output
3. Chatbot module (🔄 Partial): core/chatbot.py — CRM rules engine, rule-based LTV tier scoring, customer lookup, keyword Q&A done. Pending: Claude API Q&A, ML spending prediction, UI integration
Architecture: core/ package (analytics, charts, chatbot, config, data_loader, utils) + api/main.py (FastAPI backend with REST endpoints).
Data is in dataset/ — 7 CSVs: customers, sessions, events, orders, order_items, products, reviews.
Important: keep all project output, code, and comments in English. Do not add features beyond what is asked.
Next step: 
```

---

## 当前进度快照（2026-04-11）

| 模块 | 文件 | 状态 | 说明 |
|------|------|------|------|
| 产品分析 | `pyt/customer_analytics_dashboard_enhanced.py` | ✅ 完成（整合进仪表盘） | 销量/评分排名、VADER 情感分析、词云，全部 Plotly 交互图 |
| 用户仪表盘 | `pyt/customer_analytics_dashboard_enhanced.py` | ✅ 完成（交互 HTML） | 聚类、RFM、生命周期、漏斗、留存、地理分布 |
| ~~产品模块 Plotly 化~~ | — | ✅ 完成（Phase 1） | ProductAnalysis.py 已被整合，不再单独维护 |
| 核心逻辑包重构 | `core/` | ✅ 完成 | analytics, charts, chatbot, config, data_loader, utils 六个模块 |
| CRM 建议引擎 | `core/chatbot.py` | ✅ 完成 | `get_crm_recommendation`，覆盖所有生命周期/RFM 细分 |
| LTV 评分（规则式） | `core/chatbot.py` | ✅ 完成 | `get_ltv_tier`，Bronze/Silver/Gold/Platinum |
| 客户查询接口 | `core/chatbot.py` | ✅ 完成 | `lookup_customer`，返回完整行为+分层+CRM 建议 |
| 关键词 Q&A | `core/chatbot.py` | ✅ 完成 | `answer_question`，覆盖收入/订单/客户/留存/产品 |
| FastAPI 后端 | `api/main.py` | ✅ 完成 | KPI、RFM、生命周期、产品、地理、聊天等 REST 端点 |
| Chatbot（Claude API） | `core/chatbot.py` | 🔲 待开发 | 替换关键词匹配，接入 claude-sonnet-4-6 |
| 消费潜力 ML 预测 | — | 🔲 待开发 | XGBoost/LightGBM，当前为规则式 LTV 评分 |
| Chatbot UI | — | 🔲 待开发 | 嵌入仪表盘 |

## 关键技术细节（新会话必读）

- **项目路径**：`/Users/poetryboat/cc/Customer/`（C 大写）
- **数据目录**：`dataset/`（CSV 不在 `pyt/` 里，在 `dataset/` 里）
- **核心包路径**：`core/`（analytics, charts, chatbot, config, data_loader, utils）
- **API 入口**：`api/main.py`（FastAPI，`uvicorn api.main:app --reload` 启动）
- **已安装包**：pandas, numpy, plotly, scikit-learn, faker, vaderSentiment, wordcloud（系统 Anaconda Python `/opt/anaconda3`）
- **输出文件**：`pyt/customer_analytics_dashboard_enhanced.html`（单文件独立 HTML）
- **语言要求**：整个项目英文（代码、图表标题、HTML 输出）

## 下一步优先级

1. 接入 Claude API（claude-sonnet-4-6）替换 `core/chatbot.py` 中的关键词 Q&A
2. 训练消费潜力预测 ML 模型（XGBoost/LightGBM，替换规则式 LTV 评分）
3. 开发 Chatbot UI 并嵌入仪表盘
