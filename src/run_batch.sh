#!/bin/bash
# 分批运行数据抓取

BATCH_SIZE=100
START_INDEX=${1:-0}
END_INDEX=$((START_INDEX + BATCH_SIZE))

echo "Running batch: $START_INDEX to $END_INDEX"
cd /root/.openclaw/workspace/claude-test/stock-analyzer/a_stock_analyzer/src
python3 -c "
import sys
sys.path.insert(0, '.')
from fetcher_v2 import DataFetcherV2
from config import TUSHARE_TOKEN
import pandas as pd

fetcher = DataFetcherV2(TUSHARE_TOKEN)
stocks = fetcher.get_all_stocks_basic()
print(f'Total stocks: {len(stocks)}')

# Get batch
batch = stocks.iloc[$START_INDEX:$END_INDEX]
print(f'Processing batch: {len(batch)} stocks')

# Fetch data
data = fetcher.fetch_all_data(years=3, sample=None)
print(f'Data fetched: {len(data)} stocks')
"

echo "Batch $START_INDEX-$END_INDEX completed"
