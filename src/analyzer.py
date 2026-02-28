"""
分析器模块 - 计算股票基本面指标的三年统计数据
包含均值、中位数、标准差、趋势分析等功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from config import ANALYSIS_INDICATORS, ANALYSIS_YEARS

logger = logging.getLogger(__name__)


class FundamentalAnalyzer:
    """基本面分析器类"""
    
    def __init__(self, years: int = ANALYSIS_YEARS):
        """
        初始化分析器
        
        Args:
            years: 分析的年数，默认3年
        """
        self.years = years
        self.indicators = list(ANALYSIS_INDICATORS.keys())
        logger.info(f"FundamentalAnalyzer 初始化完成，分析年限: {years}年")
    
    def analyze_stock(self, df: pd.DataFrame,
                      stock_info: Dict = None) -> Dict:
        """
        分析单只股票的基本面数据

        Args:
            df: 该股票的指标数据（日线数据或年度数据）
            stock_info: 股票基本信息

        Returns:
            分析结果字典
        """
        if df.empty:
            return self._empty_result(stock_info)

        # 判断数据类型（年度数据还是日线数据）
        date_col = "end_date" if "end_date" in df.columns else "trade_date"

        result = {
            "ts_code": df["ts_code"].iloc[0] if "ts_code" in df.columns else "",
            "data_type": "yearly" if date_col == "end_date" else "daily",
            "analysis_period": f"{self.years}年",
            "total_records": len(df),
        }

        # 添加股票基本信息
        if stock_info:
            result.update({
                "name": stock_info.get("name", ""),
                "industry": stock_info.get("industry", ""),
                "area": stock_info.get("area", ""),
                "list_date": stock_info.get("list_date", ""),
                "market": stock_info.get("market", ""),
            })

        # 转换日期
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df["year"] = df[date_col].dt.year

        # 计算整体统计值
        overall_stats = self._calculate_overall_stats(df, date_col)
        result.update(overall_stats)

        # 计算年度统计值
        yearly_stats = self._calculate_yearly_stats(df, date_col)
        result["yearly_data"] = yearly_stats

        # 计算趋势
        trends = self._calculate_trends(df)
        result["trends"] = trends

        # 计算最新值
        latest = self._get_latest_values(df, date_col)
        result["latest"] = latest

        # 价值投资评分
        score = self._calculate_value_score(result)
        result["value_score"] = score

        return result
    
    def _calculate_overall_stats(self, df: pd.DataFrame, date_col: str = "trade_date") -> Dict:
        """计算整体统计值（三年）"""
        stats = {}
        
        for indicator in self.indicators:
            if indicator not in df.columns:
                continue
            
            data = df[indicator].dropna()
            
            if len(data) == 0:
                stats[f"{indicator}_3y_mean"] = np.nan
                stats[f"{indicator}_3y_median"] = np.nan
                stats[f"{indicator}_3y_std"] = np.nan
                stats[f"{indicator}_3y_min"] = np.nan
                stats[f"{indicator}_3y_max"] = np.nan
                stats[f"{indicator}_3y_cv"] = np.nan  # 变异系数
            else:
                stats[f"{indicator}_3y_mean"] = data.mean()
                stats[f"{indicator}_3y_median"] = data.median()
                stats[f"{indicator}_3y_std"] = data.std()
                stats[f"{indicator}_3y_min"] = data.min()
                stats[f"{indicator}_3y_max"] = data.max()
                # 变异系数 = 标准差 / 均值，用于衡量波动性
                stats[f"{indicator}_3y_cv"] = data.std() / abs(data.mean()) if data.mean() != 0 else np.nan
        
        return stats
    
    def _calculate_yearly_stats(self, df: pd.DataFrame, date_col: str = "trade_date") -> Dict:
        """计算每年的统计值"""
        yearly_data = {}
        
        for year in sorted(df["year"].unique()):
            year_df = df[df["year"] == year]
            year_stats = {"record_count": len(year_df)}
            
            for indicator in self.indicators:
                if indicator not in year_df.columns:
                    continue
                
                data = year_df[indicator].dropna()
                
                if len(data) == 0:
                    year_stats[f"{indicator}_mean"] = np.nan
                    year_stats[f"{indicator}_median"] = np.nan
                else:
                    year_stats[f"{indicator}_mean"] = data.mean()
                    year_stats[f"{indicator}_median"] = data.median()
            
            yearly_data[str(year)] = year_stats
        
        return yearly_data
    
    def _calculate_trends(self, df: pd.DataFrame) -> Dict:
        """计算指标趋势"""
        trends = {}
        
        for indicator in self.indicators:
            if indicator not in df.columns:
                continue
            
            # 按年计算均值，然后判断趋势
            yearly_means = df.groupby("year")[indicator].mean()
            
            if len(yearly_means) < 2:
                trends[f"{indicator}_trend"] = "insufficient_data"
                trends[f"{indicator}_trend_slope"] = np.nan
                continue
            
            # 使用线性回归计算趋势
            x = np.arange(len(yearly_means))
            y = pd.to_numeric(yearly_means.values, errors='coerce')
            
            # 过滤掉NaN
            valid_mask = ~np.isnan(y)
            if valid_mask.sum() < 2:
                trends[f"{indicator}_trend"] = "insufficient_data"
                trends[f"{indicator}_trend_slope"] = np.nan
                continue
            
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            
            # 线性回归
            slope = np.polyfit(x_valid, y_valid, 1)[0]
            trends[f"{indicator}_trend_slope"] = slope
            
            # 判断趋势方向
            if abs(slope) < 0.01 * abs(y_valid.mean()) if not np.isnan(y_valid.mean()) and y_valid.mean() != 0 else 0.01:
                trends[f"{indicator}_trend"] = "stable"
            elif slope > 0:
                trends[f"{indicator}_trend"] = "increasing"
            else:
                trends[f"{indicator}_trend"] = "decreasing"
        
        return trends
    
    def _get_latest_values(self, df: pd.DataFrame, date_col: str = "trade_date") -> Dict:
        """获取最新值"""
        latest = {}

        # 判断数据类型
        is_yearly = date_col == "end_date"

        # 获取最新一条记录
        latest_row = df.sort_values(date_col).iloc[-1]

        latest["trade_date"] = latest_row[date_col].strftime("%Y%m%d") if pd.notna(latest_row[date_col]) else ""

        for indicator in self.indicators:
            if indicator in latest_row.index:
                latest[indicator] = latest_row[indicator]

        return latest
    
    def _calculate_value_score(self, result: Dict) -> Dict:
        """
        计算价值投资评分
        
        评分维度：
        1. 估值吸引力（PE、PB越低越好）
        2. 股息回报（股息率越高越好）
        3. 稳定性（变异系数越低越好）
        """
        score = {
            "total_score": 0,
            "max_possible_score": 100,
            "valuation_score": 0,  # 估值评分 (0-40)
            "dividend_score": 0,   # 股息评分 (0-30)
            "stability_score": 0,  # 稳定性评分 (0-30)
            "details": {}
        }
        
        # 1. 估值评分 (40分)
        pe_mean = result.get("pe_ttm_3y_mean") or result.get("pe_3y_mean")
        pb_mean = result.get("pb_3y_mean")
        
        if pe_mean is not None and not np.isnan(pe_mean):
            if pe_mean < 10:
                score["valuation_score"] += 20
                score["details"]["pe_score"] = 20
            elif pe_mean < 15:
                score["valuation_score"] += 15
                score["details"]["pe_score"] = 15
            elif pe_mean < 20:
                score["valuation_score"] += 10
                score["details"]["pe_score"] = 10
            elif pe_mean < 30:
                score["valuation_score"] += 5
                score["details"]["pe_score"] = 5
            else:
                score["details"]["pe_score"] = 0
        
        if pb_mean is not None and not np.isnan(pb_mean):
            if pb_mean < 1:
                score["valuation_score"] += 20
                score["details"]["pb_score"] = 20
            elif pb_mean < 1.5:
                score["valuation_score"] += 15
                score["details"]["pb_score"] = 15
            elif pb_mean < 2:
                score["valuation_score"] += 10
                score["details"]["pb_score"] = 10
            elif pb_mean < 3:
                score["valuation_score"] += 5
                score["details"]["pb_score"] = 5
            else:
                score["details"]["pb_score"] = 0
        
        # 2. 股息评分 (30分)
        dv_mean = result.get("dv_ttm_3y_mean") or result.get("dv_ratio_3y_mean")
        
        if dv_mean is not None and not np.isnan(dv_mean):
            if dv_mean > 5:
                score["dividend_score"] = 30
                score["details"]["dividend_score"] = 30
            elif dv_mean > 4:
                score["dividend_score"] = 25
                score["details"]["dividend_score"] = 25
            elif dv_mean > 3:
                score["dividend_score"] = 20
                score["details"]["dividend_score"] = 20
            elif dv_mean > 2:
                score["dividend_score"] = 15
                score["details"]["dividend_score"] = 15
            elif dv_mean > 1:
                score["dividend_score"] = 10
                score["details"]["dividend_score"] = 10
            else:
                score["dividend_score"] = 0
                score["details"]["dividend_score"] = 0
        
        # 3. 稳定性评分 (30分)
        pe_cv = result.get("pe_ttm_3y_cv") or result.get("pe_3y_cv")
        
        if pe_cv is not None and not np.isnan(pe_cv):
            if pe_cv < 0.1:
                score["stability_score"] = 30
                score["details"]["stability_score"] = 30
            elif pe_cv < 0.2:
                score["stability_score"] = 20
                score["details"]["stability_score"] = 20
            elif pe_cv < 0.3:
                score["stability_score"] = 10
                score["details"]["stability_score"] = 10
            else:
                score["stability_score"] = 0
                score["details"]["stability_score"] = 0
        
        # 计算总分
        score["total_score"] = score["valuation_score"] + score["dividend_score"] + score["stability_score"]
        
        # 评级
        total = score["total_score"]
        if total >= 80:
            score["rating"] = "A+"
            score["recommendation"] = "强烈推荐"
        elif total >= 70:
            score["rating"] = "A"
            score["recommendation"] = "推荐"
        elif total >= 60:
            score["rating"] = "B"
            score["recommendation"] = "关注"
        elif total >= 40:
            score["rating"] = "C"
            score["recommendation"] = "谨慎"
        else:
            score["rating"] = "D"
            score["recommendation"] = "回避"
        
        return score
    
    def _empty_result(self, stock_info: Dict = None) -> Dict:
        """返回空结果"""
        result = {
            "ts_code": stock_info.get("ts_code", "") if stock_info else "",
            "name": stock_info.get("name", "") if stock_info else "",
            "analysis_period": f"{self.years}年",
            "total_records": 0,
            "error": "无数据"
        }
        return result
    
    def analyze_all_stocks(self,
                          data_dict: Dict[str, pd.DataFrame],
                          stock_info_dict: Dict[str, Dict]) -> pd.DataFrame:
        """
        批量分析所有股票

        Args:
            data_dict: 股票代码到数据的映射字典
            stock_info_dict: 股票代码到基本信息的映射字典

        Returns:
            分析结果DataFrame
        """
        results = []
        total = len(data_dict)

        for i, (ts_code, df) in enumerate(data_dict.items()):
            if i % 100 == 0:
                logger.info(f"分析进度: {i}/{total} ({i/total*100:.1f}%)")

            stock_info = stock_info_dict.get(ts_code, {})
            result = self.analyze_stock(df, stock_info)
            results.append(result)

        # 转换为DataFrame
        results_df = self._results_to_dataframe(results)

        logger.info(f"分析完成，共 {len(results_df)} 只股票")
        return results_df
    
    def _results_to_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """将分析结果转换为DataFrame"""
        # 提取主要字段
        rows = []
        
        for result in results:
            row = {
                "股票代码": result.get("ts_code", ""),
                "股票名称": result.get("name", ""),
                "所属行业": result.get("industry", ""),
                "所在地域": result.get("area", ""),
                "上市日期": result.get("list_date", ""),
                "市场类型": result.get("market", ""),
                "分析周期": result.get("analysis_period", ""),
                "数据记录数": result.get("total_records", 0),
                
                # 最新值
                "最新日期": result.get("latest", {}).get("trade_date", ""),
                "最新PE": result.get("latest", {}).get("pe_ttm") or result.get("latest", {}).get("pe"),
                "最新PB": result.get("latest", {}).get("pb"),
                "最新股息率": result.get("latest", {}).get("dv_ttm") or result.get("latest", {}).get("dv_ratio"),
                "最新总市值(万元)": result.get("latest", {}).get("total_mv"),
                "最新流通市值(万元)": result.get("latest", {}).get("circ_mv"),
                
                # 三年均值
                "PE_3年均值": result.get("pe_ttm_3y_mean") or result.get("pe_3y_mean"),
                "PB_3年均值": result.get("pb_3y_mean"),
                "股息率_3年均值": result.get("dv_ttm_3y_mean") or result.get("dv_ratio_3y_mean"),
                "换手率_3年均值": result.get("turnover_rate_3y_mean"),
                "总市值_3年均值(万元)": result.get("total_mv_3y_mean"),
                
                # 三年中位数
                "PE_3年中位数": result.get("pe_ttm_3y_median") or result.get("pe_3y_median"),
                "PB_3年中位数": result.get("pb_3y_median"),
                "股息率_3年中位数": result.get("dv_ttm_3y_median") or result.get("dv_ratio_3y_median"),
                
                # 波动率（变异系数）
                "PE_3年变异系数": result.get("pe_ttm_3y_cv") or result.get("pe_3y_cv"),
                "PB_3年变异系数": result.get("pb_3y_cv"),
                
                # 趋势
                "PE趋势": result.get("trends", {}).get("pe_ttm_trend") or result.get("trends", {}).get("pe_trend"),
                "PB趋势": result.get("trends", {}).get("pb_trend"),
                "股息率趋势": result.get("trends", {}).get("dv_ttm_trend") or result.get("trends", {}).get("dv_ratio_trend"),
                
                # 评分
                "价值评分": result.get("value_score", {}).get("total_score", 0),
                "评级": result.get("value_score", {}).get("rating", ""),
                "投资建议": result.get("value_score", {}).get("recommendation", ""),
            }
            
            # 添加每年的数据
            yearly_data = result.get("yearly_data", {})
            for year, year_stats in yearly_data.items():
                # 股息率（优先使用dv_ttm）
                dv = year_stats.get("dv_ttm_mean") or year_stats.get("dv_ratio_mean")
                row[f"{year}年股息率"] = dv
                
                # PE（优先使用pe_ttm）
                pe = year_stats.get("pe_ttm_mean") or year_stats.get("pe_mean")
                row[f"{year}年PE"] = pe
                
                # PB
                pb = year_stats.get("pb_mean")
                row[f"{year}年PB"] = pb
                
                # 数据记录数
                row[f"{year}年记录数"] = year_stats.get("record_count", 0)
            
            rows.append(row)
        
        return pd.DataFrame(rows)


def filter_stocks_by_criteria(df: pd.DataFrame, 
                               criteria: Dict) -> pd.DataFrame:
    """
    根据条件筛选股票
    
    Args:
        df: 分析结果DataFrame
        criteria: 筛选条件字典
        
    Returns:
        筛选后的DataFrame
    """
    filtered = df.copy()
    
    # PE范围
    if "pe_max" in criteria:
        filtered = filtered[filtered["PE_3年均值"] <= criteria["pe_max"]]
    if "pe_min" in criteria:
        filtered = filtered[filtered["PE_3年均值"] >= criteria["pe_min"]]
    
    # PB范围
    if "pb_max" in criteria:
        filtered = filtered[filtered["PB_3年均值"] <= criteria["pb_max"]]
    if "pb_min" in criteria:
        filtered = filtered[filtered["PB_3年均值"] >= criteria["pb_min"]]
    
    # 股息率
    if "dv_min" in criteria:
        filtered = filtered[filtered["股息率_3年均值"] >= criteria["dv_min"]]
    
    # 价值评分
    if "min_score" in criteria:
        filtered = filtered[filtered["价值评分"] >= criteria["min_score"]]
    
    # 行业
    if "industry" in criteria:
        filtered = filtered[filtered["所属行业"].isin(criteria["industry"])]
    
    return filtered


if __name__ == "__main__":
    # 测试代码
    analyzer = FundamentalAnalyzer(years=3)
    
    # 创建测试数据
    test_data = pd.DataFrame({
        "ts_code": ["000001.SZ"] * 100,
        "trade_date": pd.date_range("20220101", periods=100, freq="3D").strftime("%Y%m%d"),
        "pe": np.random.normal(15, 3, 100),
        "pe_ttm": np.random.normal(14, 2.5, 100),
        "pb": np.random.normal(1.5, 0.3, 100),
        "dv_ratio": np.random.normal(3, 0.5, 100),
        "dv_ttm": np.random.normal(3.2, 0.4, 100),
        "turnover_rate": np.random.normal(2, 0.5, 100),
        "total_mv": np.random.normal(1000000, 200000, 100),
    })
    
    stock_info = {
        "name": "平安银行",
        "industry": "银行",
        "area": "深圳",
        "list_date": "19910403",
        "market": "主板"
    }
    
    print("测试分析...")
    result = analyzer.analyze_stock(test_data, stock_info)
    
    print("\n分析结果:")
    print(f"股票代码: {result['ts_code']}")
    print(f"股票名称: {result['name']}")
    print(f"PE 3年均值: {result.get('pe_ttm_3y_mean', result.get('pe_3y_mean')):.2f}")
    print(f"PB 3年均值: {result.get('pb_3y_mean'):.2f}")
    print(f"股息率 3年均值: {result.get('dv_ttm_3y_mean', result.get('dv_ratio_3y_mean')):.2f}%")
    print(f"价值评分: {result['value_score']['total_score']}")
    print(f"评级: {result['value_score']['rating']}")
    print(f"建议: {result['value_score']['recommendation']}")
