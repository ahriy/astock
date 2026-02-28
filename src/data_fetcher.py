"""
数据获取模块 - 封装Tushare Pro API调用
包含重试机制、频率控制和数据缓存功能
"""

import tushare as ts
import pandas as pd
import numpy as np
import time
import os
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
from config import (
    TUSHARE_TOKEN, API_CALL_DELAY, API_MAX_RETRY, API_RETRY_DELAY,
    DATA_DIR, DAILY_BASIC_FIELDS, STOCK_BASIC_FIELDS,
    FINA_INDICATOR_FIELDS, DAILY_BASIC_FIELDS_MINIMAL, ANNUAL_REPORT_DATES
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TushareDataFetcher:
    """Tushare数据获取器类"""
    
    def __init__(self, token: Optional[str] = None):
        """
        初始化数据获取器
        
        Args:
            token: Tushare Pro API Token，如果不提供则使用配置文件中的token
        """
        self.token = token or TUSHARE_TOKEN
        if not self.token or self.token == "your_token_here":
            raise ValueError("请提供有效的Tushare Pro API Token！")
        
        # 初始化pro接口
        ts.set_token(self.token)
        self.pro = ts.pro_api()
        
        # 缓存目录
        self.cache_dir = os.path.join(DATA_DIR, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 调用计数器
        self.call_count = 0
        self.last_call_time = None
        
        logger.info("TushareDataFetcher 初始化完成")
    
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
                    logger.warning(f"API返回空数据: {func_name}, args={args}, kwargs={kwargs}")
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
    
    def _get_cache_path(self, cache_name: str) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"{cache_name}.pkl")
    
    def _load_cache(self, cache_name: str) -> Optional[pd.DataFrame]:
        """从缓存加载数据"""
        cache_path = self._get_cache_path(cache_name)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"从缓存加载数据: {cache_name}")
                return data
            except Exception as e:
                logger.warning(f"缓存加载失败: {e}")
        return None
    
    def _save_cache(self, data: pd.DataFrame, cache_name: str):
        """保存数据到缓存"""
        cache_path = self._get_cache_path(cache_name)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"数据已缓存: {cache_name}")
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")
    
    def get_trade_calendar(self, start_date: str, end_date: str, 
                          exchange: str = 'SSE') -> pd.DataFrame:
        """
        获取交易日历
        
        Args:
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            exchange: 交易所代码，SSE为上交所，SZSE为深交所
            
        Returns:
            交易日历DataFrame
        """
        cache_name = f"trade_cal_{exchange}_{start_date}_{end_date}"
        cached = self._load_cache(cache_name)
        if cached is not None:
            return cached
        
        result = self._api_call_with_retry(
            self.pro.trade_cal,
            exchange=exchange,
            start_date=start_date,
            end_date=end_date,
            is_open='1'  # 只获取开市日
        )
        
        if result is not None:
            self._save_cache(result, cache_name)
        
        return result if result is not None else pd.DataFrame()
    
    def get_all_stocks(self, list_status: str = 'L') -> pd.DataFrame:
        """
        获取所有A股股票列表
        
        Args:
            list_status: 上市状态，L=上市，D=退市，P=暂停上市
            
        Returns:
            股票基本信息DataFrame
        """
        cache_name = f"stock_basic_{list_status}"
        cached = self._load_cache(cache_name)
        if cached is not None:
            return cached
        
        result = self._api_call_with_retry(
            self.pro.stock_basic,
            exchange='',
            list_status=list_status,
            fields=','.join(STOCK_BASIC_FIELDS)
        )
        
        if result is not None:
            self._save_cache(result, cache_name)
        
        return result if result is not None else pd.DataFrame()
    
    def get_daily_basic(self, ts_code: str = '', trade_date: str = '',
                       start_date: str = '', end_date: str = '') -> pd.DataFrame:
        """
        获取每日指标数据
        
        Args:
            ts_code: 股票代码（二选一）
            trade_date: 交易日期（二选一）
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            每日指标DataFrame
        """
        # 构建缓存名称
        if ts_code:
            cache_name = f"daily_basic_{ts_code}_{start_date}_{end_date}"
        else:
            cache_name = f"daily_basic_all_{trade_date}"
        
        cached = self._load_cache(cache_name)
        if cached is not None:
            return cached
        
        result = self._api_call_with_retry(
            self.pro.daily_basic,
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
            fields=','.join(DAILY_BASIC_FIELDS)
        )
        
        if result is not None:
            self._save_cache(result, cache_name)
        
        return result if result is not None else pd.DataFrame()
    
    def get_daily_basic_by_date(self, trade_date: str) -> pd.DataFrame:
        """
        获取某交易日的全部股票每日指标
        
        Args:
            trade_date: 交易日期 (YYYYMMDD)
            
        Returns:
            该交易日全部股票的每日指标
        """
        return self.get_daily_basic(trade_date=trade_date)
    
    def get_daily_basic_by_stock(self, ts_code: str,
                                  start_date: str,
                                  end_date: str,
                                  use_yearly: bool = True) -> pd.DataFrame:
        """
        获取某股票在指定日期范围内的指标数据

        Args:
            ts_code: 股票代码
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            use_yearly: 是否使用年度数据模式，True=获取年度财务指标，False=获取日线数据

        Returns:
            该股票在指定范围内的指标数据
        """
        if use_yearly:
            # 使用年度财务指标模式
            start_year = int(start_date[:4])
            end_year = int(end_date[:4])

            # 获取年度完整数据（财务指标 + 估值指标）
            result = self.get_yearly_fina_data(ts_code, start_year, end_year)

            return result
        else:
            # 使用原有日线数据模式
            return self.get_daily_basic(ts_code=ts_code,
                                       start_date=start_date,
                                       end_date=end_date)
    
    def get_bak_basic(self, trade_date: str = '', ts_code: str = '') -> pd.DataFrame:
        """
        获取备用基础列表（包含更多基本面数据）
        
        Args:
            trade_date: 交易日期
            ts_code: 股票代码
            
        Returns:
            备用基础数据DataFrame
        """
        cache_name = f"bak_basic_{trade_date}_{ts_code}"
        cached = self._load_cache(cache_name)
        if cached is not None:
            return cached
        
        result = self._api_call_with_retry(
            self.pro.bak_basic,
            trade_date=trade_date,
            ts_code=ts_code
        )
        
        if result is not None:
            self._save_cache(result, cache_name)
        
        return result if result is not None else pd.DataFrame()
    
    def get_stock_company(self, ts_code: str = '', 
                         exchange: str = '') -> pd.DataFrame:
        """
        获取上市公司基本信息
        
        Args:
            ts_code: 股票代码
            exchange: 交易所代码
            
        Returns:
            上市公司信息DataFrame
        """
        result = self._api_call_with_retry(
            self.pro.stock_company,
            ts_code=ts_code,
            exchange=exchange
        )
        return result if result is not None else pd.DataFrame()
    
    def get_fina_indicator(self, ts_code: str,
                          start_date: str = '',
                          end_date: str = '',
                          period: str = '年报') -> pd.DataFrame:
        """
        获取财务指标数据（支持年度数据筛选）

        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            period: 报告期类型，'年报'/'中报'/'季报'，默认为'年报'

        Returns:
            财务指标DataFrame
        """
        cache_name = f"fina_indicator_{ts_code}_{start_date}_{end_date}_{period}"
        cached = self._load_cache(cache_name)
        if cached is not None:
            return cached

        result = self._api_call_with_retry(
            self.pro.fina_indicator,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )

        if result is not None:
            # 筛选年报数据（period字段末尾为'1231'或12月31日）
            if period == '年报':
                # 筛选年报数据：end_date末尾为1231，或者period字段包含年报标识
                result = result[result['end_date'].str.endswith('1231', na=False)].copy()
            elif period == '中报':
                # 中报：0630
                result = result[result['end_date'].str.endswith('0630', na=False)].copy()
            elif period == '一季报':
                result = result[result['end_date'].str.endswith('0331', na=False)].copy()
            elif period == '三季报':
                result = result[result['end_date'].str.endswith('0930', na=False)].copy()

            self._save_cache(result, cache_name)

        return result if result is not None else pd.DataFrame()

    def _generate_annual_dates(self, start_year: int, end_year: int,
                              date_type: str = 'default') -> List[str]:
        """
        生成年报时间点日期列表

        Args:
            start_year: 开始年份
            end_year: 结束年份
            date_type: 日期类型，'default'=12月31日，'alternative'=4月30日

        Returns:
            日期字符串列表（YYYYMMDD格式）
        """
        date_suffix = ANNUAL_REPORT_DATES.get(date_type, ANNUAL_REPORT_DATES['default'])
        dates = []
        for year in range(start_year, end_year + 1):
            dates.append(f"{year}{date_suffix}")
        return dates

    def get_yearly_basic(self, ts_code: str,
                        start_year: int,
                        end_year: int,
                        date_type: str = 'default') -> pd.DataFrame:
        """
        获取年度基础数据（每年12月31日的数据）

        Args:
            ts_code: 股票代码
            start_year: 开始年份
            end_year: 结束年份
            date_type: 日期类型，'default'=12月31日，'alternative'=4月30日

        Returns:
            年度基础数据DataFrame，包含估值指标（pe, pb, dv_ratio等）
        """
        # 生成要获取的日期列表
        annual_dates = self._generate_annual_dates(start_year, end_year, date_type)

        cache_name = f"yearly_basic_{ts_code}_{start_year}_{end_year}_{date_type}"
        cached = self._load_cache(cache_name)
        if cached is not None:
            return cached

        all_data = []
        for date in annual_dates:
            # 获取该日期的日线基础数据
            daily_data = self._api_call_with_retry(
                self.pro.daily_basic,
                ts_code=ts_code,
                trade_date=date,
                fields=','.join(DAILY_BASIC_FIELDS_MINIMAL)
            )

            if daily_data is not None and not daily_data.empty:
                all_data.append(daily_data)
            # 如果没有数据，跳过该年份（不尝试查找附近交易日）

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            # 按日期排序
            result = result.sort_values('trade_date').reset_index(drop=True)
            self._save_cache(result, cache_name)
            return result

        return pd.DataFrame()

    def get_yearly_fina_data(self, ts_code: str,
                            start_year: int,
                            end_year: int) -> pd.DataFrame:
        """
        获取年度完整数据（财务指标 + 估值指标）

        Args:
            ts_code: 股票代码
            start_year: 开始年份
            end_year: 结束年份

        Returns:
            合并后的年度数据DataFrame，格式与daily_basic兼容
        """
        cache_name = f"yearly_fina_{ts_code}_{start_year}_{end_year}"
        cached = self._load_cache(cache_name)
        if cached is not None:
            return cached

        # 获取年报财务指标
        start_date = f"{start_year}0101"
        end_date = f"{end_year}1231"
        fina_data = self.get_fina_indicator(ts_code, start_date, end_date, period='年报')

        if fina_data.empty:
            logger.warning(f"股票 {ts_code} 未获取到财务指标数据")
            return pd.DataFrame()

        # 获取年报时间点的估值数据
        basic_data = self.get_yearly_basic(ts_code, start_year, end_year)

        if basic_data.empty:
            logger.warning(f"股票 {ts_code} 未获取到估值数据")
            # 返回只有财务指标的数据
            return fina_data

        # 合并数据：基于年份匹配
        merged_data = self._merge_yearly_data(fina_data, basic_data)

        if not merged_data.empty:
            self._save_cache(merged_data, cache_name)

        return merged_data

    def _merge_yearly_data(self, fina_data: pd.DataFrame,
                          basic_data: pd.DataFrame) -> pd.DataFrame:
        """
        合并财务指标数据和估值数据

        Args:
            fina_data: 财务指标数据
            basic_data: 估值数据（daily_basic格式）

        Returns:
            合并后的DataFrame
        """
        result_rows = []

        for _, fina_row in fina_data.iterrows():
            end_date = fina_row['end_date']  # 格式: YYYYMMDD
            year = end_date[:4]

            # 在basic_data中找到对应年份的数据（12月31日）
            target_month_day = ANNUAL_REPORT_DATES['default']  # 默认1231
            target_date = f"{year}{target_month_day}"

            # 查找匹配的basic数据（同一年份即可）
            matching_basic = None
            for _, basic_row in basic_data.iterrows():
                basic_date = basic_row['trade_date']
                basic_year = basic_date[:4]
                # 同一年份
                if basic_year == year:
                    matching_basic = basic_row
                    break

            if matching_basic is not None:
                # 合并两行数据
                merged_row = {**fina_row.to_dict(), **matching_basic.to_dict()}
                # 添加trade_date字段以便兼容
                merged_row['trade_date'] = merged_row.get('trade_date', end_date)
                result_rows.append(merged_row)
            else:
                # 没有找到匹配的估值数据，只使用财务指标
                row_dict = fina_row.to_dict()
                row_dict['trade_date'] = end_date
                # 填充缺失的估值字段为NaN
                for field in DAILY_BASIC_FIELDS_MINIMAL:
                    if field not in row_dict:
                        row_dict[field] = np.nan
                result_rows.append(row_dict)

        if result_rows:
            return pd.DataFrame(result_rows)

        return pd.DataFrame()

    def get_latest_trade_dates(self, n_days: int = 5) -> List[str]:
        """
        获取最近N个交易日
        
        Args:
            n_days: 需要获取的交易日数量
            
        Returns:
            最近N个交易日的日期列表（YYYYMMDD格式）
        """
        # 获取最近30天的交易日历
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        
        trade_cal = self.get_trade_calendar(
            start_date=start_date.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d')
        )
        
        if trade_cal.empty:
            return []
        
        # 按日期排序，取最近的N个
        trade_dates = sorted(trade_cal['cal_date'].tolist(), reverse=True)
        return trade_dates[:n_days]
    
    def clear_cache(self):
        """清除所有缓存数据"""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info("缓存已清除")
    
    def get_call_stats(self) -> Dict:
        """获取API调用统计信息"""
        return {
            "total_calls": self.call_count,
            "last_call_time": self.last_call_time
        }


if __name__ == "__main__":
    # 测试代码
    try:
        fetcher = TushareDataFetcher()
        
        # 测试获取股票列表
        print("获取股票列表...")
        stocks = fetcher.get_all_stocks()
        print(f"获取到 {len(stocks)} 只股票")
        print(stocks.head())
        
        # 测试获取交易日历
        print("\n获取交易日历...")
        trade_dates = fetcher.get_trade_calendar('20240101', '20240131')
        print(f"获取到 {len(trade_dates)} 个交易日")
        print(trade_dates.head())
        
    except Exception as e:
        print(f"测试失败: {e}")
