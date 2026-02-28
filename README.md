# A股基本面分析器

基于Tushare Pro API的A股价值投资分析工具，自动抓取全量A股数据，计算最近三年的基本面指标（PE、PB、股息率等），并生成价值投资评分和综合分析报告。

## 核心特性

- **断点保护**: 每100只股票自动保存进度，支持断点续传，程序中断后可从上次位置继续
- **单版本输出**: 只保留最新分析结果，固定文件名便于自动化处理
- **三年历史分析**: 按年份分别显示每年的PE、PB、股息率等指标
- **价值评分**: 基于估值、股息、稳定性三维度计算评分（A+到D评级）
- **数据验证**: 多重数据质量检查，确保分析结果准确性

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行测试

```bash
python test.py
```

### 3. 运行分析

```bash
cd src
python main.py --sample 100   # 测试模式（100只股票）
python main.py               # 完整分析（全部A股，约2-3小时）
```

### 4. 查看结果

分析结果保存在 `output` 目录：
- `a_stock_analysis.xlsx` - 全部股票分析结果
- `comprehensive_report.xlsx` - 综合报告（多工作表）
- `summary_report.txt` - 汇总统计信息

## 断点续传

程序会自动保存进度到 `data/checkpoint.pkl`，中断后重新运行会自动从上次位置继续。

```bash
# 从断点继续
python main.py

# 重新开始（忽略断点）
python main.py --restart

# 查看当前进度
python main.py --progress
```

## 项目结构

```
a_stock_analyzer/
├── src/
│   ├── config.py          # 配置文件（Token、分析年限等）
│   ├── main.py            # 主程序入口（含断点保护逻辑）
│   ├── data_fetcher.py    # 数据获取模块
│   ├── data_validator.py  # 数据验证模块
│   ├── analyzer.py        # 分析模块（计算指标、评分）
│   └── exporter.py        # 导出模块
├── data/                  # 数据缓存目录（含断点文件checkpoint.pkl）
├── output/                # 分析结果输出目录
├── logs/                  # 日志目录
├── requirements.txt       # 依赖列表
├── README.md             # 本文件
├── QUICKSTART.md         # 快速开始指南
└── test.py               # 测试脚本
```

## 核心模块说明

### main.py
- 主程序入口，包含 `AStockAnalyzer` 类
- 实现断点保护：`CheckpointManager` 类管理进度保存/恢复
- 每 `CHECKPOINT_INTERVAL`（默认100）只股票自动保存
- 支持信号处理（Ctrl+C安全退出）

### data_fetcher.py
- `TushareDataFetcher` 类封装Tushare Pro API
- 包含重试机制、频率控制、数据缓存
- 核心方法：
  - `get_all_stocks()` - 获取股票列表
  - `get_daily_basic_by_stock()` - 获取单只股票历史数据
  - `get_trade_calendar()` - 获取交易日历

### data_validator.py
- `DataValidator` 类负责数据质量检查
- 异常值检测（IQR方法）
- 数据清洗（处理极端值、重复数据）

### analyzer.py
- `FundamentalAnalyzer` 类计算指标和评分
- 核心方法：
  - `analyze_stock()` - 分析单只股票
  - `_calculate_yearly_stats()` - 计算年度统计
  - `_calculate_value_score()` - 计算价值评分
- 输出列包含每年的PE、PB、股息率（如：2023年股息率、2024年股息率等）

### exporter.py
- `DataExporter` 类负责结果导出
- 固定文件名输出（单版本模式）
- 支持Excel、CSV格式

## 配置说明（config.py）

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| TUSHARE_TOKEN | Tushare Pro API Token | 需填写 |
| ANALYSIS_YEARS | 分析年限 | 3 |
| API_CALL_DELAY | API调用间隔（秒） | 1.2 |
| CHECKPOINT_INTERVAL | 断点保存间隔（只） | 100 |

## 输出列说明

### 基本信息
- 股票代码、股票名称、所属行业、所在地域

### 最新值
- 最新日期、最新PE、最新PB、最新股息率

### 年度数据（按年份分别列出）
- 2023年股息率、2024年股息率、2025年股息率、2026年股息率
- 2023年PE、2024年PE、2025年PE、2026年PE
- 2023年PB、2024年PB、2025年PB、2026年PB

### 统计值
- PE_3年均值、PB_3年均值、股息率_3年均值
- PE_3年中位数、PB_3年中位数、股息率_3年中位数

### 趋势与评分
- PE趋势、PB趋势、股息率趋势
- 价值评分、评级、投资建议

## 价值评分标准

| 总分 | 评级 | 建议 |
|------|------|------|
| 80-100 | A+ | 强烈推荐 |
| 70-79 | A | 推荐 |
| 60-69 | B | 关注 |
| 40-59 | C | 谨慎 |
| 0-39 | D | 回避 |

## 命令行参数

```bash
python main.py [选项]

选项:
  --token TOKEN      Tushare Pro API Token
  --years YEARS      分析年限（默认3年）
  --sample N         抽样数量（用于测试）
  --restart          重新开始（忽略断点）
  --progress         只显示进度信息
```

## 注意事项

1. **积分要求**: `daily_basic`接口需要至少2000积分
2. **运行时间**: 全量分析（5000+只股票）约需2-3小时
3. **断点文件**: 存储在 `data/checkpoint.pkl`，可手动删除以重新开始

## 许可证

MIT License

## 免责声明

本工具仅供学习研究使用，不构成投资建议。投资有风险，入市需谨慎。
