# Customer Intelligence Platform — 完整产品文档

**版本**: v1.1  
**日期**: 2026-04-11  
**项目性质**: 零售全渠道客户行为分析与关系管理平台

---

## 目录

1. [项目背景与研究问题](#1-项目背景与研究问题)
2. [数据资产](#2-数据资产)
3. [系统架构总览](#3-系统架构总览)
4. [已完成模块](#4-已完成模块)
   - 4.1 产品分析模块（ProductAnalysis.py）
   - 4.2 用户分析仪表盘（customer_analytics_dashboard_enhanced.py）
5. [待开发模块](#5-待开发模块)
   - 5.1 Chatbot 模块
6. [Dashboard 页面规划](#6-dashboard-页面规划)
7. [技术栈](#7-技术栈)
8. [文件目录结构](#8-文件目录结构)
9. [开发路线图](#9-开发路线图)
10. [附录：字段说明](#10-附录字段说明)

---

## 1. 项目背景与研究问题

**研究问题**：如何利用点击流数据（Clickstream）、交易记录（Transaction Records）和客户相关信息（Customer Information），跨线上与线下零售环境理解并建模客户行为？

**核心分析目标**：

| 分析维度 | 说明 |
|----------|------|
| 访问频率 | 客户多久回访一次（Session 复购率、留存率） |
| 停留时长 | 每次会话的行为深度（页面浏览 → 加购 → 结账漏斗） |
| 购买行为 | 购买了什么、消费金额、折扣敏感度 |
| 情感与评价 | 评论情感分析、评分趋势、产品口碑 |
| 行为演化 | 客户生命周期阶段识别与动态追踪 |

**最终交付物**：一个集成式分析仪表盘 + Chatbot，为 CRM 运营提供实时洞察与行动建议。

---

## 2. 数据资产

数据存放于 `dataset/` 文件夹，共 7 张 CSV 表：

### 2.1 数据表说明

| 文件名 | 核心字段 | 说明 |
|--------|----------|------|
| `customers.csv` | customer_id, name, email, country, age, signup_date, marketing_opt_in | 客户基础信息 |
| `sessions.csv` | session_id, customer_id, start_time, device, source, country | 用户会话记录（浏览行为） |
| `events.csv` | event_id, session_id, timestamp, event_type, product_id, qty, cart_size, payment, discount_pct, amount_usd | 会话内事件流（page_view / add_to_cart / checkout / purchase） |
| `orders.csv` | order_id, customer_id, order_time, payment_method, discount_pct, subtotal_usd, total_usd, country, device, source | 订单主表 |
| `order_items.csv` | order_id, product_id, unit_price_usd, quantity, line_total_usd | 订单明细（商品级） |
| `products.csv` | product_id, category, name, price_usd, cost_usd, margin_usd | 商品基础信息 |
| `reviews.csv` | review_id, order_id, product_id, rating, review_text, review_time | 用户评价（含文本） |

### 2.2 数据关系

```
customers ──< sessions ──< events
     │
     └──< orders ──< order_items >── products
              │
              └──< reviews >── products
```

---

## 3. 系统架构总览

```
┌─────────────────────────────────────────────────────────┐
│                   Customer Intelligence Platform         │
├──────────────┬──────────────────┬───────────────────────┤
│  产品模块     │    用户模块       │      Chatbot 模块      │
│              │                  │                        │
│ • 销量 Rank  │ • 聚类用户画像    │ • 自然语言问答          │
│ • 评分 Rank  │ • RFM 分析        │ • 消费潜力预测          │
│ • 情感分析   │ • 生命周期阶段    │ • CRM 行动建议          │
│ • 词云       │ • 转化漏斗        │                        │
│ • 用户画像   │ • 留存矩阵        │                        │
└──────────────┴──────────────────┴───────────────────────┘
                       ▼
              ┌─────────────────┐
              │  Plotly 交互仪表盘│
              │  (HTML 单页输出)  │
              └─────────────────┘
```

---

## 4. 已完成模块

### 4.1 产品分析模块

**状态**: ✅ 已完成并整合进主仪表盘（Phase 1，2026-04-11）

`ProductAnalysis.py` 原为独立静态图模块，现已完全迁移至 `customer_analytics_dashboard_enhanced.py`，不再单独维护。

#### 当前功能（均为 Plotly 交互图，输出至统一 HTML）

| 功能 | 实现函数 | 图表类型 |
|------|----------|----------|
| 销量 Top 10 排名 | `make_product_sales_rank` | 横向条形图（按类目着色） |
| 评分 Top 10 排名 | `make_product_rating_rank` | 横向条形图（最少 5 条评论过滤） |
| 类目情感分析 | `make_sentiment_stacked_bar` | 堆叠条形图（VADER，Positive / Neutral / Negative） |
| 评分 × 销售额气泡图 | `make_category_bubble` | 气泡散点图（已有） |
| 词云（按类目） | `generate_wordcloud_section_html` | 所有类目自动生成，base64 PNG 嵌入 HTML |

#### 情感分析逻辑（已升级）

使用 `vaderSentiment` 库，compound score ≥ 0.05 → Positive，≤ -0.05 → Negative，其余 → Neutral。入口函数：`sentiment_product_metrics(data)`。

---

### 4.2 用户分析仪表盘（`customer_analytics_dashboard_enhanced.py`）

**状态**: ✅ 核心逻辑已完成，输出为交互式 HTML

#### 功能清单

**A. 客户画像构建（`build_customer_profile`）**

| 特征类型 | 具体特征 |
|----------|----------|
| 基础人口 | 年龄段（6档）、性别推断（姓名匹配）、国家、地区 |
| 会话行为 | 会话次数、首/末会话时间、主要来源、主要设备、来源多样性 |
| 事件漏斗 | page_views、add_to_cart、checkout、purchase 各阶段计数 |
| 转化率 | page→cart rate、cart→checkout rate、checkout→purchase rate |
| 订单价值 | 订单数、总收入、平均客单价、平均折扣率 |
| 活跃度 | active_days（首末会话跨度）、recency_days（距今天数） |
| 评价行为 | 评论数、平均评分、评论率（评论数/订单数） |

**B. RFM 分析（`add_rfm_segments`）**

将客户按 Recency / Frequency / Monetary 各维度打分（1-5分），组合成以下细分：

| RFM 细分 | 特征 |
|----------|------|
| Champions | R≥4, F≥4, M≥4，最高价值客户 |
| Loyal Customers | R≥4, F≥3，高频近期活跃 |
| At-Risk High Value | R≤2, F≥4，曾高频但已流失风险 |
| Lost Low Value | R≤2, F≤2, M≤2，已流失低价值 |
| Price / Casual Shoppers | F≤2, M≤2，偶发性低消费 |
| Potential Loyalists | 其余，潜力待激活 |

**C. 生命周期阶段（`add_lifecycle_stage`）**

| 阶段 | 判断条件 |
|------|----------|
| New Visitor | 0订单 & 最近30天内注册 |
| Browsing Prospect | 0订单 & 有会话记录 |
| New Buyer | 1笔订单 & 最近45天内 |
| Active Repeat Buyer | ≥2笔订单 & 最近60天内 |
| Cooling Down | ≥1笔订单 & 60-120天未活跃 |
| Dormant Customer | ≥1笔订单 & 120天以上未活跃 |

**D. 用户聚类（`assign_cluster_personas`）**

- **算法**：KMeans（自动选优 k=3~6，Silhouette Score 评估）
- **特征**（17维，经标准化）：session_count_log、revenue_log、转化率序列、avg_discount_pct、avg_rating、active_days、recency_days、source/device 多样性、age、marketing_opt_in
- **PCA 降维**：2D 可视化
- **自动命名的 Persona**：

| Persona | 特征 |
|---------|------|
| High-Value Repeat Buyers | 高收入 & 高订单频次 |
| Heavy Browsers, Low Conversion | 高会话 & 低转化 |
| Promotion Sensitive | 高折扣偏好 |
| Dormant / Churn Risk | 高 recency（长期未活跃） |
| High Intent, High Conversion | 高转化率 |
| Steady Mainstream | 其余均衡群体 |

**E. 转化漏斗（`funnel_metrics`）**

- 总体漏斗：page_view → add_to_cart → checkout → purchase（会话级）
- 按设备分组：desktop / mobile / tablet 各阶段转化率
- 按来源分组：organic / direct / paid / social / email / referral 各阶段转化率

**F. 地理分析（`geography_metrics`）**

覆盖 17 个国家/地区（含 region 归类），输出：会话数、活跃用户、订单数、收入、AOV、转化率。

**G. 月度趋势（`monthly_overview`）**

- 月度会话数、活跃用户、订单数、收入、AOV、转化率
- 3个月滚动平均收入
- 月均评分与评论量

**H. 留存矩阵（`retention_matrix`）**

- 按注册月份（Cohort）追踪 12 个月留存率
- 支持 sessions 或 orders 两种活跃口径

**I. 产品指标（`product_metrics`）**

- 类目月度收入/销量/毛利趋势
- 类目整体表现（收入、销量、毛利、评分）
- Top 12 商品（按收入排序）

---

## 5. 待开发模块

### 5.1 Chatbot 模块

**状态**: 🔄 部分完成（规则引擎已完成，Claude API 问答待接入）

已完成（`core/chatbot.py`）：
- ✅ CRM 建议规则引擎（`get_crm_recommendation`，覆盖所有生命周期/RFM 细分）
- ✅ LTV 评分与潜力分层（`get_ltv_tier`，规则式，Bronze/Silver/Gold/Platinum）
- ✅ 客户画像查询（`lookup_customer`，返回完整行为+分层+CRM 建议）
- ✅ 关键词 Q&A（`answer_question`，覆盖收入/订单/客户/留存/产品等问题类型）
- 🔲 Claude API 问答接入（替换关键词匹配，文件注释标记为 Phase 2）
- 🔲 消费潜力 ML 预测（XGBoost / LightGBM，当前为规则式评分）
- 🔲 Chatbot UI 嵌入仪表盘

#### 5.1.1 问答子系统（Query Answering）

支持业务人员用自然语言查询数据洞察。

**预期支持的问题类型**：

| 类型 | 示例问题 |
|------|----------|
| 产品销量 | "近7日销量 Top 5 是哪些产品？" |
| 产品评分 | "Electronics 类目评分最低的 3 个产品是？" |
| 用户行为 | "上个月新用户转化率是多少？" |
| 地区分析 | "哪个国家的平均客单价最高？" |
| 留存/流失 | "目前有多少 Cooling Down 客户？" |
| 趋势查询 | "最近 3 个月收入趋势如何？" |

**技术方案建议**：

```
用户输入（自然语言）
        ↓
意图识别（关键词匹配 / Claude API 解析）
        ↓
参数提取（时间范围、产品类目、指标名称、Top-N 等）
        ↓
SQL / Pandas 查询执行
        ↓
结果格式化 → 文字 + 可选图表
```

**推荐实现**：使用 Claude API（claude-sonnet-4-6）将自然语言转换为结构化查询参数，再执行 Pandas 计算。

#### 5.1.2 消费潜力预测（Spending Potential Prediction）

对已有客户群，预测其未来消费能力，用于精准营销和资源分配。

**输入特征**（来自 `build_customer_profile`）：

| 特征 | 说明 |
|------|------|
| rfm_score | RFM 综合评分 |
| lifecycle_stage | 当前生命周期阶段 |
| session_count | 历史会话次数 |
| page_to_cart_rate | 浏览转加购率 |
| avg_order_value | 历史平均客单价 |
| avg_discount_pct | 折扣敏感度 |
| active_days | 活跃跨度 |
| recency_days | 最近活跃距今天数 |
| marketing_opt_in | 是否接受营销 |
| cluster_name | 所属 Persona 群体 |

**预测目标**：
- `high_potential`：二分类（高/低消费潜力）
- `predicted_ltv_tier`：多分类（Bronze / Silver / Gold / Platinum）

**建议算法**：XGBoost 或 LightGBM（基于历史订单数据训练）

**输出形式**：
- 单个客户的潜力评分（0-100）
- Chatbot 返回："该用户属于 High-Value Repeat Buyers，预测消费潜力等级为 Gold，建议推送高端产品优惠券。"

#### 5.1.3 CRM 行动建议（CRM Recommendations）

根据客户当前状态，Chatbot 自动给出运营建议。

| 生命周期 / RFM 细分 | CRM 建议 |
|--------------------|---------|
| New Visitor | 发送欢迎邮件 + 首单优惠码 |
| Browsing Prospect | 推送浏览商品的限时折扣提醒 |
| New Buyer | 发送产品使用指南 + 交叉销售推荐 |
| Active Repeat Buyer | 推荐会员计划 / 积分奖励 |
| Cooling Down | 发送 "我们想念你" 召回邮件 + 专属折扣 |
| Dormant Customer | 高价值：电话/重度激活；低价值：低成本 EDM |
| Champions | VIP 专属活动邀请 + 新品优先体验 |
| At-Risk High Value | 紧急召回：定向优惠 + 客服主动联系 |
| Promotion Sensitive | 推送闪购/折扣活动，避免全价推送 |

---

## 6. Dashboard 页面规划

完整仪表盘采用 Plotly 交互式 HTML 单页输出，分以下模块/标签：

### Tab 1：概览（Overview）
- KPI 卡片：总客户数、总收入、平均 AOV、整体转化率
- 月度收入趋势折线图（含 3 月滚动均值）
- 月度活跃用户 & 会话量趋势

### Tab 2：产品模块（Products）
- 销量 Top 10 排名（交互可切换：7日 / 30日 / 全量）
- 评分 Top 10 排名
- 类目情感分析堆叠图（Positive / Neutral / Negative 占比）
- 词云（按类目筛选，嵌入 HTML）
- 评分 × 销售额气泡散点图
- 类目月度趋势（Revenue / Units / Gross Margin）

### Tab 3：用户模块（Customers）
- 聚类 Persona 概览卡片（地区、来源、转化率、平均年龄等）
- PCA 2D 聚类散点图
- RFM 分层客户数 & 收入贡献条形图
- 生命周期阶段分布漏斗
- 加购行为漏斗（page_view → add_to_cart → checkout → purchase）
  - 总体漏斗
  - 按设备拆分
  - 按流量来源拆分
- 人口统计：地区 × 性别 × 年龄 Sunburst 图
- 人口统计：年龄段性别对比（蝴蝶图）
- 地理分布地图（国家 Choropleth）

### Tab 4：留存分析（Retention）
- 月度留存矩阵热力图（Cohort Analysis，12个月）
- 可切换：基于 Sessions / Orders 口径

### Tab 5：Chatbot
- 问答输入框
- 历史对话记录展示
- 消费潜力评估结果展示（客户 ID 输入 → 潜力评分 + 建议）

---

## 7. 技术栈

| 层次 | 技术 |
|------|------|
| 数据处理 | pandas, numpy |
| 机器学习 | scikit-learn（KMeans, PCA, StandardScaler, Silhouette）|
| 可视化 | plotly（交互式 HTML）, matplotlib/seaborn（静态图）, wordcloud |
| 情感分析 | 规则词典（现有）→ 建议升级 TextBlob / VADER |
| 预测建模 | XGBoost / LightGBM（待开发） |
| Chatbot 后端 | Claude API（claude-sonnet-4-6）+ pandas 查询执行层 |
| 前端输出 | Plotly HTML 单页 + 可选 Streamlit / Dash |
| 开发环境 | Python 3.14, virtualenv（.venv） |

---

## 8. 文件目录结构

```
Customer/
├── dataset/                          # 原始数据（CSV 读取路径，非 pyt/）
│   ├── customers.csv
│   ├── sessions.csv
│   ├── events.csv
│   ├── orders.csv
│   ├── order_items.csv
│   ├── products.csv
│   └── reviews.csv
│
├── core/                             # ✅ 核心业务逻辑包（Phase 2 重构产物）
│   ├── __init__.py
│   ├── analytics.py                  # ✅ 客户画像、RFM、聚类、漏斗、留存、产品指标
│   ├── charts.py                     # ✅ Plotly 图表构建 + 词云生成
│   ├── chatbot.py                    # ✅ CRM 规则、LTV 评分、客户查询、关键词 Q&A
│   ├── config.py                     # ✅ 全局配置（路径、常量、颜色）
│   ├── data_loader.py                # ✅ CSV 数据加载
│   └── utils.py                      # ✅ 工具函数（格式化、地区映射、性别推断）
│
├── api/                              # ✅ FastAPI 后端（Phase 3 先行实现）
│   ├── __init__.py
│   └── main.py                       # ✅ REST API（健康检查、KPI、聊天、客户、产品等端点）
│
├── pyt/                              # 独立脚本（早期版本）
│   ├── customer_analytics_dashboard_enhanced.py  # ✅ 主仪表盘（产品 + 用户模块合并）
│   ├── customer_analytics_dashboard_enhanced.html  # ✅ 输出 HTML（运行上方脚本生成）
│   └── ProductAnalysis.py            # ⚠️ 已被整合，仅留存参考，不再维护
│
├── PROJECT_DOCUMENT.md               # 本文档
└── RESUME_PROMPT.md                  # 新会话重启 Prompt
```

---

## 9. 开发路线图

### Phase 1：整合与统一（✅ 已完成，2026-04-11）
- [x] 产品分析核心逻辑（ProductAnalysis.py）
- [x] 用户聚类 + 生命周期 + RFM 分析
- [x] 交互式 HTML 仪表盘输出
- [x] 将产品模块迁移至 Plotly 并合并进主仪表盘
- [x] 情感分析升级（VADER）
- [x] 词云参数化（支持所有类目动态生成，base64 嵌入 HTML）

### Phase 2：Chatbot 开发（🔄 进行中）
- [ ] 自然语言问答（Claude API claude-sonnet-4-6 接入，替换当前关键词匹配）
- [ ] 消费潜力预测 ML 模型（XGBoost / LightGBM，当前为规则式 LTV 评分）
- [x] CRM 建议规则引擎（`core/chatbot.py` — `get_crm_recommendation`）
- [x] 客户画像查询与 LTV 评分（`core/chatbot.py` — `lookup_customer`, `get_ltv_tier`）
- [x] 关键词 Q&A 基础框架（`core/chatbot.py` — `answer_question`）
- [ ] Chatbot UI 嵌入仪表盘

### Phase 3：系统集成与优化（🔄 部分完成）
- [x] FastAPI 后端（`api/main.py`，含 KPI / 客户 / 产品 / 聊天端点）
- [x] 核心逻辑包化重构（`core/` — analytics, charts, chatbot, config, data_loader, utils）
- [ ] 数据管道自动化（定时刷新 CSV → HTML）
- [ ] 部署（Streamlit Cloud / 本地服务器）

---

## 10. 附录：字段说明

### 事件类型（event_type）

| 值 | 含义 |
|----|------|
| `page_view` | 浏览产品页 |
| `add_to_cart` | 加入购物车 |
| `checkout` | 进入结账流程 |
| `purchase` | 完成购买 |

### 流量来源（source）

| 值 | 含义 |
|----|------|
| `organic` | 自然搜索 |
| `direct` | 直接访问 |
| `paid` | 付费广告 |
| `social` | 社交媒体 |
| `email` | 邮件营销 |
| `referral` | 外链引荐 |

### RFM 评分维度

| 维度 | 含义 | 数据来源 |
|------|------|----------|
| R（Recency） | 最近一次购买距今天数（越小越好） | `orders.order_time` |
| F（Frequency） | 购买总次数 | `orders.order_id` count |
| M（Monetary） | 购买总金额 | `orders.total_usd` sum |

---

*文档由 Claude Code 基于现有代码自动生成，人工确认后有效。如有模块更新请同步修改本文档。*
