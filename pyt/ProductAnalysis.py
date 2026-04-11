import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import matplotlib.gridspec as gridspec


# Load data
customers = pd.read_csv('customers.csv')
events = pd.read_csv('events.csv')
orders = pd.read_csv('orders.csv')
sessions = pd.read_csv('sessions.csv')
reviews = pd.read_csv('reviews.csv')
products = pd.read_csv('products.csv')
order_items = pd.read_csv('order_items.csv')

# aet theme for a professional look
sns.set_theme(style="whitegrid", palette="muted")

# calculate sales volume and revenue
product_sales = order_items.groupby('product_id').agg(
    total_qty=('quantity', 'sum'),
    total_revenue=('line_total_usd', 'sum')
).reset_index()

# Calculate average ratings and review density
product_ratings = reviews.groupby('product_id').agg(
    avg_rating=('rating', 'mean'),
    review_count=('rating', 'count')
).reset_index()

# merge into a master analysis dataframe
product_master = products.merge(product_sales, on='product_id', how='left').fillna(0)
product_master = product_master.merge(product_ratings, on='product_id', how='left').fillna(0)

# 2. SENTIMENT ANALYSIS
def analyze_sentiment(text):
    text = str(text).lower()
    pos = ['excellent', 'great', 'good', 'love', 'perfect', 'recommend', 'value']
    neg = ['bad', 'poor', 'disappointed', 'worst', 'broken', 'issue']
    score = sum(1 for w in pos if w in text) - sum(1 for w in neg if w in text)
    return 'Positive' if score > 0 else ('Negative' if score < 0 else 'Neutral')

reviews['sentiment'] = reviews['review_text'].apply(analyze_sentiment)
cat_sentiment = reviews.merge(products[['product_id', 'category']], on='product_id')
sentiment_pivot = cat_sentiment.groupby(['category', 'sentiment']).size().unstack(fill_value=0)
sentiment_pct = sentiment_pivot.div(sentiment_pivot.sum(axis=1), axis=0) * 100

# 3. CREATE INTEGRATED DASHBOARD
fig = plt.figure(figsize=(24, 18))
gs = gridspec.GridSpec(3, 2, figure=fig)

#sales volume
ax1 = fig.add_subplot(gs[0, 0])
top_sales = product_master.sort_values('total_qty', ascending=False).head(10)
sns.barplot(data=top_sales, x='total_qty', y='name', hue='name', palette='viridis', legend=False, ax=ax1)
ax1.set_title('Top 10 Products by Sales Volume', fontweight='bold')

# credible ratings
ax2 = fig.add_subplot(gs[0, 1])
top_rated = product_master[product_master['review_count'] >= 5].sort_values('avg_rating', ascending=False).head(10)
sns.barplot(data=top_rated, x='avg_rating', y='name', hue='name', palette='magma', legend=False, ax=ax2)
ax2.set_title('Top 10 Rated Products (Min. 5 Reviews)', fontweight='bold')

# sentiment distribution
ax3 = fig.add_subplot(gs[1, 0])
sentiment_pct.plot(kind='barh', stacked=True, color=['#e74c3c', '#95a5a6', '#2ecc71'], ax=ax3)
ax3.set_title('Sentiment Distribution Across Categories (%)', fontweight='bold')

# revenue vs. quality correlation
ax4 = fig.add_subplot(gs[1, 1])
sns.scatterplot(data=product_master, x='avg_rating', y='total_revenue', hue='category', size='review_count', sizes=(20, 400), alpha=0.6, ax=ax4)
ax4.set_title('Revenue vs. Rating (Size = Review Density)', fontweight='bold')

# category word clouds
def generate_wc(category, subplot_idx):
    text = " ".join(reviews.merge(products, on='product_id')[reviews.merge(products, on='product_id')['category'] == category]['review_text'])
    wc = WordCloud(width=600, height=300, background_color='white', colormap='cool').generate(text)
    ax = fig.add_subplot(gs[2, subplot_idx])
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Keyword Trends: {category}', fontweight='bold')

generate_wc('Electronics', 0)
generate_wc('Beauty', 1)
# set any category if u need 

plt.tight_layout()
plt.savefig('dashboard_for_preview.png')
