"""
A股数据获取器 V2.0
功能：获取全部A股基础数据并保存
- 收盘价
- PE（市盈率）
- PB（市净率）
- ROE（净资产收益率）
- 毛利率（gross_margin）
- 资产负债率（debt_to_assets）
"""

import os
import time
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Set, List
import pandas as pd
import numpy as np
import tushare as ts
from config import (
    TUSHARE_TOKEN,
    OUTPUT_DIR,
    API_CALL_DELAY,
    API_MAX_RETRY,
    API_RETRY_DELAY,
    LOG_CONFIG
)

# 配置日志
logging.basicConfig(
    level=LOG_CONFIG["level"],
    format=LOG_CONFIG["format"],
    handlers=[
        logging.FileHandler(LOG_CONFIG["file"], encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# V2.0 数据字段配置
# 标准完整财务指标字段（从fina_indicator接口获取）
FINA_INDICATOR_V2_FIELDS = [
    "ts_code",          # TS股票代码
    "end_date",         # 报告期
    # 利润表核心字段
    "eps",              # 每股收益
    "dt_eps",           # 扣除每股收益
    "total_revenue_ps", # 每股营业总收入
    "revenue_ps",       # 每股营业收入
    "profit_dedt",      # 扣除非经常损益后的净利润
    # 资产负债表字段
    "bps",              # 每股净资产
    "capital_rese_ps",  # 每股资本公积
    "surplus_rese_ps",  # 每股盈余公积
    "undist_profit_ps", # 每股未分配利润
    # 盈利能力指标
    "roe",              # 净资产收益率
    "roe_waa",          # 加权平均净资产收益率
    "roe_dt",           # 扣除非经常损益后的净资产收益率
    "roe_yearly",       # 年化净资产收益率
    "netprofit_margin", # 销售净利率
    "gross_margin",     # 毛利率
    # 运营效率指标
    "assets_turn",      # 总资产周转率
    "ca_turn",          # 流动资产周转率
    # 偿债能力指标
    "debt_to_assets",   # 资产负债率
    "current_ratio",    # 流动比率
    "quick_ratio",      # 速动比率
    "cash_ratio",       # 保守速动比率
    # 现金流指标
    "op_income",        # 经营活动净收益
    "fcff",             # 企业自由现金流量
    "fcfe",             # 股权自由现金流量
]

# 股票基本信息字段
STOCK_BASIC_FIELDS = [
    "ts_code",      # TS代码
    "symbol",       # 股票代码
    "name",         # 股票名称
    "area",         # 所在地域
    "industry",     # 所属行业
    "market",       # 市场类型
    "exchange",     # 交易所代码
    "list_date",    # 上市日期
    "is_hs",        # 是否沪深港通标的
]

# 日线基础指标（用于获取PE、PB等估值数据）
DAILY_BASIC_FIELDS = [
    "ts_code",
    "trade_date",
    "close",
    "pe",
    "pe_ttm",
    "pb",
    "ps",
    "ps_ttm",
    "dv_ratio",
    "dv_ttm",
    "total_mv",
    "circ_mv",
]


class DataFetcherV2:
    """V2.0 数据获取器 - 获取全量A股原始数据"""

    def __init__(self, token: str = None, raw_data_dir: str = None):
        """
        初始化数据获取器

        Args:
            token: Tushare Pro API Token
            raw_data_dir: 原始数据保存目录
        """
        self.token = token or TUSHARE_TOKEN
        if not self.token or self.token == "your_token_here":
            raise ValueError("请提供有效的Tushare Pro API Token！")

        # 初始化pro接口
        ts.set_token(self.token)
        self.pro = ts.pro_api()

        # 设置原始数据目录
        if raw_data_dir is None:
            # 默认使用项目根目录下的raw_data
            project_root = Path(__file__).parent.parent
            raw_data_dir = project_root / "raw_data"
        self.raw_data_dir = Path(raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        # 调用计数器和频率控制
        self.call_count = 0
        self.last_call_time = None

        # Checkpoint 相关
        self.checkpoint_interval = 100  # 每处理100只股票保存一次checkpoint
        self.batch_save_interval = 500  # 每处理500只股票保存一次数据
        self.processed_stocks: Set[str] = set()  # 已处理的股票代码集合
        self.current_date = datetime.now().strftime('%Y%m%d')
        self.checkpoint_file = self.raw_data_dir / f"checkpoint_{self.current_date}.json"
        self.output_file = self.raw_data_dir / f"all_stocks_{self.current_date}.csv"
        self.failed_file = self.raw_data_dir / f"failed_stocks_{self.current_date}.csv"

        # 加载已处理的股票（如果有checkpoint）
        self._load_checkpoint_if_exists()

        logger.info(f"DataFetcherV2 初始化完成，原始数据目录: {self.raw_data_dir}")

    def _load_checkpoint_if_exists(self):
        """如果存在checkpoint文件，加载已处理的股票列表"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                    self.processed_stocks = set(checkpoint_data.get('processed_stocks', []))
                    logger.info(f"从checkpoint恢复: 已处理 {len(self.processed_stocks)} 只股票")
            except Exception as e:
                logger.warning(f"加载checkpoint失败: {e}，将从头开始")
                self.processed_stocks = set()
        else:
            self.processed_stocks = set()

    def _save_checkpoint(self, current_position: int):
        """
        保存checkpoint到文件

        Args:
            current_position: 当前处理位置（股票索引）
        """
        try:
            checkpoint_data = {
                'date': self.current_date,
                'processed_stocks': list(self.processed_stocks),
                'current_position': current_position,
                'timestamp': datetime.now().isoformat()
            }
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            logger.debug(f"Checkpoint已保存: 位置 {current_position}, 已处理 {len(self.processed_stocks)} 只股票")
        except Exception as e:
            logger.warning(f"保存checkpoint失败: {e}")

    def _append_batch_to_file(self, batch_data: List[Dict], write_header: bool):
        """
        将一批数据追加保存到CSV文件

        Args:
            batch_data: 要保存的数据列表
            write_header: 是否写入表头（第一批需要写表头）
        """
        if not batch_data:
            return

        try:
            batch_df = pd.DataFrame(batch_data)
            batch_df.to_csv(
                self.output_file,
                mode='a',  # 追加模式
                header=write_header,  # 只有第一批写表头
                index=False,
                encoding='utf-8-sig'
            )
            logger.info(f"已保存 {len(batch_data)} 只股票到 {self.output_file}")
        except Exception as e:
            logger.error(f"保存批次数据失败: {e}")

    def _save_failed_stock(self, ts_code: str, reason: str):
        """
        记录失败的股票到文件

        Args:
            ts_code: 股票代码
            reason: 失败原因
        """
        try:
            # 检查文件是否存在，决定是否写表头
            write_header = not self.failed_file.exists()

            with open(self.failed_file, 'a', encoding='utf-8-sig') as f:
                if write_header:
                    f.write('ts_code,reason,timestamp\n')
                f.write(f'{ts_code},{reason},{datetime.now().isoformat()}\n')
        except Exception as e:
            logger.warning(f"记录失败股票时出错: {e}")

    def _cleanup_checkpoint(self):
        """清理checkpoint文件"""
        if self.checkpoint_file.exists():
            try:
                self.checkpoint_file.unlink()
                logger.info(f"Checkpoint文件已删除: {self.checkpoint_file}")
            except Exception as e:
                logger.warning(f"删除checkpoint文件失败: {e}")

    def _api_call_with_retry(self, func, *args, **kwargs) -> Optional[pd.DataFrame]:
        """
        带重试机制的API调用

        Args:
            func: API函数
            *args, **kwargs: 函数参数

        Returns:
            API返回的DataFrame，失败返回None
        """
        for attempt in range(API_MAX_RETRY):
            try:
                # 频率控制
                self._rate_limit()

                # 执行API调用
                result = func(*args, **kwargs)

                # 记录调用
                self.call_count += 1
                self.last_call_time = time.time()

                # 检查结果
                if result is None or result.empty:
                    func_name = getattr(func, '__name__', str(func))
                    logger.warning(f"API返回空数据: {func_name}")
                    return pd.DataFrame()

                return result

            except Exception as e:
                error_msg = str(e)
                logger.warning(f"API调用失败 (尝试 {attempt + 1}/{API_MAX_RETRY}): {error_msg}")

                # 检查是否是积分不足或权限问题
                if "积分" in error_msg or "权限" in error_msg:
                    logger.error(f"积分或权限不足，无法继续: {error_msg}")
                    raise

                # 检查是否是频率限制
                if "频率" in error_msg or "limit" in error_msg.lower():
                    wait_time = API_RETRY_DELAY * (attempt + 1)
                    logger.info(f"触发频率限制，等待 {wait_time} 秒...")
                    time.sleep(wait_time)
                else:
                    time.sleep(API_RETRY_DELAY)

                if attempt == API_MAX_RETRY - 1:
                    func_name = getattr(func, '__name__', str(func))
                    logger.error(f"API调用最终失败: {func_name}")
                    return None

        return None

    def _rate_limit(self):
        """频率控制 - 确保API调用间隔"""
        if self.last_call_time is not None:
            elapsed = time.time() - self.last_call_time
            if elapsed < API_CALL_DELAY:
                sleep_time = API_CALL_DELAY - elapsed
                time.sleep(sleep_time)

    def _validate_and_clean_financial_data(self, data: Dict) -> Dict:
        """
        验证和清洗财务数据

        处理数据单位问题和异常值：
        - 毛利率(gross_margin): Tushare返回的是小数形式(如0.15表示15%)，需要转换为百分比
        - ROE: 应该在-50%到50%之间
        - 毛利率: 应该在0%到100%之间
        - 资产负债率: 应该在0%到100%之间
        - 销售净利率: 应该在-50%到50%之间
        - 总资产周转率: 应该在0到10之间
        - 流动资产周转率: 应该在0到20之间

        Args:
            data: 原始数据字典

        Returns:
            清洗后的数据字典
        """
        cleaned = data.copy()

        # 1. 毛利率单位转换和验证
        gross_margin = cleaned.get('gross_margin', np.nan)
        if pd.notna(gross_margin):
            if gross_margin > 1:
                if gross_margin > 100:
                    cleaned['gross_margin'] = None
                else:
                    cleaned['gross_margin'] = gross_margin
            else:
                cleaned['gross_margin'] = gross_margin * 100
            if pd.notna(cleaned['gross_margin']) and (cleaned['gross_margin'] < 0 or cleaned['gross_margin'] > 100):
                cleaned['gross_margin'] = None

        # 2. ROE验证：应该在-50%到50%之间
        for roe_field in ['roe', 'roe_waa', 'roe_dt', 'roe_yearly']:
            roe_value = cleaned.get(roe_field, np.nan)
            if pd.notna(roe_value):
                if roe_value < -50 or roe_value > 50:
                    cleaned[roe_field] = None

        # 3. 资产负债率验证：应该在0%到100%之间
        debt_to_assets = cleaned.get('debt_to_assets', np.nan)
        if pd.notna(debt_to_assets):
            if debt_to_assets > 1:
                if debt_to_assets > 100:
                    cleaned['debt_to_assets'] = None
            else:
                cleaned['debt_to_assets'] = debt_to_assets * 100
            if pd.notna(cleaned['debt_to_assets']) and (cleaned['debt_to_assets'] < 0 or cleaned['debt_to_assets'] > 100):
                cleaned['debt_to_assets'] = None

        # 4. 销售净利率验证：应该在-50%到50%之间
        netprofit_margin = cleaned.get('netprofit_margin', np.nan)
        if pd.notna(netprofit_margin):
            if netprofit_margin > 1:
                if netprofit_margin > 100:
                    cleaned['netprofit_margin'] = None
                else:
                    cleaned['netprofit_margin'] = netprofit_margin
            else:
                cleaned['netprofit_margin'] = netprofit_margin * 100
            if pd.notna(cleaned['netprofit_margin']) and (cleaned['netprofit_margin'] < -50 or cleaned['netprofit_margin'] > 50):
                cleaned['netprofit_margin'] = None

        # 5. 总资产周转率验证：应该在0到10之间
        assets_turn = cleaned.get('assets_turn', np.nan)
        if pd.notna(assets_turn):
            if assets_turn < 0 or assets_turn > 10:
                cleaned['assets_turn'] = None

        # 6. 流动资产周转率验证：应该在0到20之间
        ca_turn = cleaned.get('ca_turn', np.nan)
        if pd.notna(ca_turn):
            if ca_turn < 0 or ca_turn > 20:
                cleaned['ca_turn'] = None

        # 7. 每股指标验证（EPS、BPS等）：应该在合理范围内
        for per_share_field in ['eps', 'dt_eps', 'bps', 'total_revenue_ps', 'revenue_ps', 'capital_rese_ps', 'surplus_rese_ps', 'undist_profit_ps']:
            value = cleaned.get(per_share_field, np.nan)
            if pd.notna(value):
                # 每股数据通常在-100到1000之间（极端情况除外）
                if value < -100 or value > 1000:
                    cleaned[per_share_field] = None

        # 8. 利润验证：应该在合理范围内
        profit_dedt = cleaned.get('profit_dedt', np.nan)
        if pd.notna(profit_dedt):
            # 扣除非经常损益后的净利润，单位可能是元，允许很大的负值和正值
            # 但如果超过1万亿，可能是数据问题
            if abs(profit_dedt) > 1e12:
                cleaned['profit_dedt'] = None

        return cleaned

    def get_all_stocks_basic(self) -> pd.DataFrame:
        """
        获取全部A股基本信息

        Returns:
            股票基本信息DataFrame
        """
        logger.info("正在获取全部A股基本信息...")
        result = self._api_call_with_retry(
            self.pro.stock_basic,
            exchange='',
            list_status='L',
            fields=','.join(STOCK_BASIC_FIELDS)
        )

        if result is not None and not result.empty:
            logger.info(f"成功获取 {len(result)} 只A股基本信息")
        else:
            logger.error("获取A股基本信息失败")
            return pd.DataFrame()

        return result

    def get_financial_indicators(self, ts_code: str, years: int = 5) -> pd.DataFrame:
        """
        获取单只股票的财务指标数据

        Args:
            ts_code: 股票代码
            years: 获取最近几年的年报数据

        Returns:
            财务指标DataFrame
        """
        current_year = datetime.now().year
        start_date = f"{current_year - years}0101"
        end_date = f"{current_year}1231"

        result = self._api_call_with_retry(
            self.pro.fina_indicator,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )

        if result is None or result.empty:
            return pd.DataFrame()

        # 筛选年报数据（end_date末尾为1231）
        result = result[result['end_date'].str.endswith('1231', na=False)].copy()
        result = result.sort_values('end_date', ascending=False)

        return result

    def get_latest_valuation(self, ts_code: str) -> Dict:
        """
        获取股票的最新估值数据

        Args:
            ts_code: 股票代码

        Returns:
            估值数据字典
        """
        # 获取最近交易日的数据
        trade_date = datetime.now().strftime('%Y%m%d')
        result = self._api_call_with_retry(
            self.pro.daily_basic,
            ts_code=ts_code,
            trade_date=trade_date,
            fields=','.join(DAILY_BASIC_FIELDS)
        )

        # 如果当天没有数据，尝试前几天
        if result is None or result.empty:
            for i in range(1, 10):
                date = (datetime.now() - timedelta(days=i)).strftime('%Y%m%d')
                result = self._api_call_with_retry(
                    self.pro.daily_basic,
                    ts_code=ts_code,
                    trade_date=date,
                    fields=','.join(DAILY_BASIC_FIELDS)
                )
                if result is not None and not result.empty:
                    break

        if result is None or result.empty:
            return {}

        return result.iloc[0].to_dict()

    def calculate_growth_metrics(self, df: pd.DataFrame) -> Dict:
        """
        计算增长指标（5年EPS增长、5年销售增长）

        Args:
            df: 财务指标DataFrame

        Returns:
            增长指标字典
        """
        if df.empty or len(df) < 2:
            return {
                "eps_5y_cagr": None,
                "revenue_5y_cagr": None,
                "data_years": 0
            }

        # 按年份排序（从旧到新）
        df = df.sort_values('end_date').reset_index(drop=True)

        # 计算CAGR（复合年增长率）
        def calc_cagr(start_value, end_value, years):
            if start_value is None or end_value is None or start_value <= 0 or years <= 0:
                return None
            try:
                return (float(end_value) / float(start_value)) ** (1 / years) - 1
            except:
                return None

        growth_metrics = {}
        data_years = len(df)

        # 计算EPS增长率
        if 'eps' in df.columns:
            eps_earliest = df['eps'].iloc[0]
            eps_latest = df['eps'].iloc[-1]
            growth_metrics["eps_5y_cagr"] = calc_cagr(eps_earliest, eps_latest, data_years - 1)

        # 计算收入增长率（使用total_revenue_ps或revenue_ps）
        revenue_col = None
        for col in ['revenue_ps', 'total_revenue_ps', 'operating_revenue_ps']:
            if col in df.columns:
                revenue_col = col
                break

        if revenue_col:
            revenue_earliest = df[revenue_col].iloc[0]
            revenue_latest = df[revenue_col].iloc[-1]
            growth_metrics["revenue_5y_cagr"] = calc_cagr(revenue_earliest, revenue_latest, data_years - 1)

        growth_metrics["data_years"] = data_years

        return growth_metrics

    def fetch_all_data(self, years: int = 5, sample: int = None, progress_callback=None, resume: bool = True) -> pd.DataFrame:
        """
        获取全部A股的完整数据

        Args:
            years: 财务数据年限
            sample: 只获取前N只股票（用于测试）
            progress_callback: 进度回调函数
            resume: 是否从checkpoint恢复（默认True）

        Returns:
            完整数据DataFrame
        """
        # 获取股票列表
        stocks_df = self.get_all_stocks_basic()
        if stocks_df.empty:
            logger.error("无法获取股票列表")
            return pd.DataFrame()

        total_stocks = len(stocks_df)

        # 如果指定了sample参数，只获取前N只股票
        if sample is not None and sample > 0:
            stocks_df = stocks_df.head(sample)
            total_stocks = len(stocks_df)
            logger.info(f"样本模式：只获取前 {sample} 只股票的数据...")

        # 过滤已处理的股票（断点续传）
        if resume and self.processed_stocks:
            stocks_df = stocks_df[~stocks_df['ts_code'].isin(self.processed_stocks)].reset_index(drop=True)
            skipped_count = len(self.processed_stocks)
            logger.info(f"断点恢复: 跳过已处理的 {skipped_count} 只股票，剩余 {len(stocks_df)} 只待处理")

        total_stocks = len(stocks_df)

        if total_stocks == 0:
            logger.info("没有需要处理的股票（可能已全部完成）")
            return pd.DataFrame()

        logger.info(f"开始获取 {total_stocks} 只股票的数据...")

        # 批次处理变量
        batch_data = []  # 当前批次的数据
        batch_count = 0  # 批次计数
        processed_in_run = 0  # 本次运行处理的数量
        output_file_exists = self.output_file.exists()  # 检查输出文件是否已存在

        for idx, stock in stocks_df.iterrows():
            ts_code = stock['ts_code']
            current_progress = idx + 1

            if progress_callback:
                progress_callback(current_progress, total_stocks, ts_code)

            if current_progress % 100 == 0 or current_progress == 1:
                logger.info(f"进度: {current_progress}/{total_stocks} ({current_progress/total_stocks*100:.1f}%)")

            # 获取财务指标
            fina_df = self.get_financial_indicators(ts_code, years)

            if fina_df.empty:
                logger.debug(f"{ts_code} 无财务指标数据")
                self._save_failed_stock(ts_code, "无财务指标数据")
                # 即使失败也要记录到已处理列表
                self.processed_stocks.add(ts_code)
                # 定期保存checkpoint
                if current_progress % self.checkpoint_interval == 0:
                    self._save_checkpoint(current_progress)
                continue

            # 获取最新财务数据
            latest_fina = fina_df.iloc[0].to_dict()

            # 计算增长指标
            growth = self.calculate_growth_metrics(fina_df)

            # 获取最新估值数据
            valuation = self.get_latest_valuation(ts_code)

            # 合并数据
            stock_data = {
                'ts_code': ts_code,
                'symbol': stock.get('symbol', ''),
                'name': stock.get('name', ''),
                'area': stock.get('area', ''),
                'industry': stock.get('industry', ''),
                'market': stock.get('market', ''),
                'exchange': stock.get('exchange', ''),
                'list_date': stock.get('list_date', ''),
                'is_hs': stock.get('is_hs', ''),

                # 最新财务指标
                'end_date': latest_fina.get('end_date', ''),
                # 利润表核心字段
                'eps': latest_fina.get('eps', np.nan),
                'dt_eps': latest_fina.get('dt_eps', np.nan),
                'total_revenue_ps': latest_fina.get('total_revenue_ps', np.nan),
                'revenue_ps': latest_fina.get('revenue_ps', np.nan),
                'profit_dedt': latest_fina.get('profit_dedt', np.nan),
                # 资产负债表字段
                'bps': latest_fina.get('bps', np.nan),
                'capital_rese_ps': latest_fina.get('capital_rese_ps', np.nan),
                'surplus_rese_ps': latest_fina.get('surplus_rese_ps', np.nan),
                'undist_profit_ps': latest_fina.get('undist_profit_ps', np.nan),
                # 盈利能力指标
                'roe': latest_fina.get('roe', np.nan),
                'roe_waa': latest_fina.get('roe_waa', np.nan),
                'roe_dt': latest_fina.get('roe_dt', np.nan),
                'roe_yearly': latest_fina.get('roe_yearly', np.nan),
                'netprofit_margin': latest_fina.get('netprofit_margin', np.nan),
                'gross_margin': latest_fina.get('gross_margin', np.nan),
                # 运营效率指标
                'assets_turn': latest_fina.get('assets_turn', np.nan),
                'ca_turn': latest_fina.get('ca_turn', np.nan),
                # 偿债能力指标
                'debt_to_assets': latest_fina.get('debt_to_assets', np.nan),
                'current_ratio': latest_fina.get('current_ratio', np.nan),
                'quick_ratio': latest_fina.get('quick_ratio', np.nan),
                'cash_ratio': latest_fina.get('cash_ratio', np.nan),
                # 现金流指标
                'op_income': latest_fina.get('op_income', np.nan),
                'fcff': latest_fina.get('fcff', np.nan),
                'fcfe': latest_fina.get('fcfe', np.nan),

                # 增长指标
                'eps_5y_cagr': growth.get('eps_5y_cagr', np.nan),
                'revenue_5y_cagr': growth.get('revenue_5y_cagr', np.nan),
                'data_years': growth.get('data_years', 0),

                # 最新估值数据
                'trade_date': valuation.get('trade_date', ''),
                'close': valuation.get('close', np.nan),
                'pe': valuation.get('pe', np.nan),
                'pe_ttm': valuation.get('pe_ttm', np.nan),
                'pb': valuation.get('pb', np.nan),
                'ps': valuation.get('ps', np.nan),
                'ps_ttm': valuation.get('ps_ttm', np.nan),
                'dv_ratio': valuation.get('dv_ratio', np.nan),
                'dv_ttm': valuation.get('dv_ttm', np.nan),
                'total_mv': valuation.get('total_mv', np.nan),
                'circ_mv': valuation.get('circ_mv', np.nan),
            }

            # 应用数据清洗和验证
            stock_data = self._validate_and_clean_financial_data(stock_data)

            batch_data.append(stock_data)
            self.processed_stocks.add(ts_code)
            processed_in_run += 1

            # 每处理checkpoint_interval只股票，保存checkpoint
            if current_progress % self.checkpoint_interval == 0:
                self._save_checkpoint(current_progress)

            # 每处理batch_save_interval只股票，保存一批数据到文件
            if len(batch_data) >= self.batch_save_interval:
                # 第一批且文件不存在时需要写表头
                write_header = not output_file_exists
                self._append_batch_to_file(batch_data, write_header)
                output_file_exists = True  # 后续批次不需要写表头
                batch_data = []  # 清空批次
                batch_count += 1

        # 保存最后一批数据
        if batch_data:
            write_header = not output_file_exists
            self._append_batch_to_file(batch_data, write_header)

        # 全部处理完成后，删除checkpoint文件
        if processed_in_run > 0:
            self._cleanup_checkpoint()
            logger.info("数据处理完成，checkpoint已清理")

        # 读取完整数据返回（用于向后兼容）
        try:
            result_df = pd.read_csv(self.output_file, encoding='utf-8-sig')
        except Exception as e:
            logger.warning(f"读取输出文件失败: {e}，返回空DataFrame")
            result_df = pd.DataFrame()

        logger.info(f"数据获取完成: 本次成功处理 {processed_in_run} 只股票")
        return result_df

    def save_raw_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        保存原始数据到CSV文件

        Args:
            df: 数据DataFrame
            filename: 文件名（不含路径和扩展名）

        Returns:
            保存的文件路径
        """
        if df.empty:
            logger.error("数据为空，无法保存")
            return ""

        if filename is None:
            filename = f"all_stocks_{datetime.now().strftime('%Y%m%d')}"

        filepath = self.raw_data_dir / f"{filename}.csv"
        df.to_csv(filepath, index=False, encoding='utf-8-sig')

        logger.info(f"原始数据已保存: {filepath}")
        return str(filepath)

    def fetch_and_save(self, years: int = 5, sample: int = None, progress_callback=None) -> str:
        """
        获取数据并保存到文件（一步完成）

        Args:
            years: 财务数据年限
            sample: 只获取前N只股票（用于测试）
            progress_callback: 进度回调函数

        Returns:
            保存的文件路径
        """
        df = self.fetch_all_data(years, sample, progress_callback)
        return self.save_raw_data(df)

    def get_call_stats(self) -> Dict:
        """获取API调用统计信息"""
        return {
            "total_calls": self.call_count,
            "last_call_time": self.last_call_time
        }


if __name__ == "__main__":
    # 测试代码
    def test_progress(current, total, ts_code):
        if current % 50 == 0:
            print(f"进度: {current}/{total} - {ts_code}")

    try:
        fetcher = DataFetcherV2()
        print("测试数据获取...")

        # 测试获取单只股票数据
        test_df = fetcher.get_financial_indicators("000001.SZ", years=3)
        if not test_df.empty:
            print(f"\n测试股票财务数据:")
            print(test_df[['end_date', 'roe', 'gross_margin', 'eps']].head())

        # 测试增长指标计算
        growth = fetcher.calculate_growth_metrics(test_df)
        print(f"\n增长指标: {growth}")

        # 测试估值数据
        valuation = fetcher.get_latest_valuation("000001.SZ")
        print(f"\n估值数据: PE={valuation.get('pe')}, PB={valuation.get('pb')}")

    except Exception as e:
        print(f"测试失败: {e}")
