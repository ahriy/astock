"""
配置文件 - A股基本面分析器
用户需要在此填入自己的Tushare Pro API Token
"""

import os

# ==================== 用户配置区域 ====================

# Tushare Pro API Token
# 请从 https://tushare.pro/register 注册获取
# 免费用户有积分限制，建议至少获取2000积分以使用daily_basic接口
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "31e5536e4d0ffbfb47a74e9832fd35c711fdaa2405bec6559b62d22d")

# 数据时间范围配置
# 分析最近N年的数据
ANALYSIS_YEARS = 3

# API调用配置
# 免费用户每分钟限制60次调用，每次间隔至少1秒
API_CALL_DELAY = 0.5  # 每次API调用间隔（秒）
API_MAX_RETRY = 5     # 最大重试次数
API_RETRY_DELAY = 2   # 重试间隔（秒）

# 断点保护配置
CHECKPOINT_INTERVAL = 100  # 每处理多少只股票保存一次断点

# 数据存储路径
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")

# 确保目录存在
for dir_path in [DATA_DIR, OUTPUT_DIR, LOG_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 关键指标字段配置
# 从daily_basic接口获取的字段
DAILY_BASIC_FIELDS = [
    "ts_code",        # TS股票代码
    "trade_date",     # 交易日期
    "close",          # 当日收盘价
    "turnover_rate",  # 换手率（%）
    "turnover_rate_f",# 换手率（自由流通股）
    "volume_ratio",   # 量比
    "pe",             # 市盈率（总市值/净利润）
    "pe_ttm",         # 市盈率（TTM）
    "pb",             # 市净率（总市值/净资产）
    "ps",             # 市销率
    "ps_ttm",         # 市销率（TTM）
    "dv_ratio",       # 股息率（%）
    "dv_ttm",         # 股息率（TTM）（%）
    "total_share",    # 总股本（万股）
    "float_share",    # 流通股本（万股）
    "free_share",     # 自由流通股本（万）
    "total_mv",       # 总市值（万元）
    "circ_mv",        # 流通市值（万元）
]

# 股票基本信息字段
STOCK_BASIC_FIELDS = [
    "ts_code",      # TS代码
    "symbol",       # 股票代码
    "name",         # 股票名称
    "area",         # 所在地域
    "industry",     # 所属行业
    "fullname",     # 股票全称
    "enname",       # 英文全称
    "cnspell",      # 拼音缩写
    "market",       # 市场类型
    "exchange",     # 交易所代码
    "curr_type",    # 交易货币
    "list_status",  # 上市状态
    "list_date",    # 上市日期
    "delist_date",  # 退市日期
    "is_hs",        # 是否沪深港通标的
]

# 年度财务指标字段（从fina_indicator接口获取）
# 注意：该接口提供季度数据，我们筛选年报数据（period参数）
FINA_INDICATOR_FIELDS = [
    "ts_code",          # TS股票代码
    "ann_date",         # 公告日期
    "end_date",         # 报告期
    "eps",              # 每股收益
    "dt_eps",           # 稀释每股收益
    "total_revenue_ps", # 每股营业总收入
    "revenue_ps",       # 每股营业收入
    "capital_rese_ps",  # 每股资本公积
    "surplus_rese_ps",  # 每股盈余公积
    "undist_profit_ps", # 每股未分配利润
    "extra_item",       # 非经常性损益
    "profit_dedt",      # 扣除非经常性损益后的净利润
    "gross_margin",     # 毛利
    "current_ratio",    # 流动比率
    "quick_ratio",      # 速动比率
    "cash_ratio",       # 保守速动比率
    "ar_turn",          # 应收账款周转率
    "ca_turn",          # 流动资产周转率
    "fa_turn",          # 固定资产周转率
    "assets_turn",      # 总资产周转率
    "op_income",        # 经营活动净收益
    "valuechange_income",# 价值变动净收益
    "interst_income",   # 利息费用
    "daa",              # 固定资产折旧、油气资产折耗、生产性生物资产折旧
    "fcff",             # 企业自由现金流量
    "fcfe",             # 股权自由现金流量
]

# 日线行情基础字段（用于补充财务指标中不包含的估值数据）
DAILY_BASIC_FIELDS_MINIMAL = [
    "ts_code",        # TS股票代码
    "trade_date",     # 交易日期
    "close",          # 当日收盘价
    "pe",             # 市盈率
    "pe_ttm",         # 市盈率TTM
    "pb",             # 市净率
    "ps",             # 市销率
    "ps_ttm",         # 市销率TTM
    "dv_ratio",       # 股息率（%）
    "dv_ttm",         # 股息率TTM（%）
    "total_mv",       # 总市值（万元）
    "circ_mv",        # 流通市值（万元）
]

# 年报发布时间点配置（用于获取年报发布后的数据）
ANNUAL_REPORT_DATES = {
    "default": "1231",  # 默认每年12月31日（年末）
    "alternative": "0430",  # 备选时间点4月30日（年报披露截止日）
}

# 分析指标配置 - 用于计算三年统计值
ANALYSIS_INDICATORS = {
    # 估值指标
    "pe": {"name": "市盈率", "unit": "倍", "lower_is_better": True, "threshold": 50},
    "pe_ttm": {"name": "市盈率TTM", "unit": "倍", "lower_is_better": True, "threshold": 50},
    "pb": {"name": "市净率", "unit": "倍", "lower_is_better": True, "threshold": 10},
    "ps": {"name": "市销率", "unit": "倍", "lower_is_better": True, "threshold": 20},
    "ps_ttm": {"name": "市销率TTM", "unit": "倍", "lower_is_better": True, "threshold": 20},
    
    # 股息指标
    "dv_ratio": {"name": "股息率", "unit": "%", "lower_is_better": False, "threshold": 3},
    "dv_ttm": {"name": "股息率TTM", "unit": "%", "lower_is_better": False, "threshold": 3},
    
    # 流动性指标
    "turnover_rate": {"name": "换手率", "unit": "%", "lower_is_better": None, "threshold": None},
    "volume_ratio": {"name": "量比", "unit": "", "lower_is_better": None, "threshold": None},
    
    # 规模指标
    "total_mv": {"name": "总市值", "unit": "万元", "lower_is_better": None, "threshold": None},
    "circ_mv": {"name": "流通市值", "unit": "万元", "lower_is_better": None, "threshold": None},
}

# 数据质量检查配置
DATA_QUALITY_CONFIG = {
    "min_data_points_per_year": 200,  # 每年最少需要的数据点（交易日约250天）
    "max_pe_threshold": 1000,         # PE最大值阈值，超过视为异常
    "min_pe_threshold": 0,            # PE最小值阈值
    "max_pb_threshold": 100,          # PB最大值阈值
    "min_pb_threshold": 0,            # PB最小值阈值
    "max_dv_ratio": 50,               # 股息率最大值阈值
    "outlier_std_threshold": 3,       # 异常值判断标准差倍数
}

# 日志配置
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": os.path.join(LOG_DIR, "analyzer.log"),
}
