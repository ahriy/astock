"""
数据验证模块 - 确保数据准确性和完整性
包含数据质量检查、异常值检测和数据清洗功能
支持日频数据和年度财务数据的验证
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
from config import DATA_QUALITY_CONFIG, ANALYSIS_INDICATORS, ANALYSIS_YEARS

logger = logging.getLogger(__name__)


class DataValidator:
    """数据验证器类"""

    def __init__(self):
        self.config = DATA_QUALITY_CONFIG
        self.quality_report = {}
        logger.info("DataValidator 初始化完成")
    
    def validate_daily_basic(self, df: pd.DataFrame, 
                            ts_code: str = '', 
                            year: int = None) -> Dict:
        """
        验证每日指标数据的完整性和准确性
        
        Args:
            df: 每日指标DataFrame
            ts_code: 股票代码（用于日志记录）
            year: 年份（用于日志记录）
            
        Returns:
            验证报告字典
        """
        report = {
            "ts_code": ts_code,
            "year": year,
            "total_records": len(df),
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "missing_fields": [],
            "outliers": {},
            "data_quality_score": 100
        }
        
        if df.empty:
            report["is_valid"] = False
            report["errors"].append("数据为空")
            report["data_quality_score"] = 0
            return report
        
        # 1. 检查必要字段
        required_fields = ["ts_code", "trade_date", "pe", "pb", "dv_ratio"]
        for field in required_fields:
            if field not in df.columns:
                report["missing_fields"].append(field)
                report["warnings"].append(f"缺少字段: {field}")
        
        # 2. 检查数据量
        if year and len(df) < self.config["min_data_points_per_year"]:
            report["warnings"].append(
                f"数据量不足: {len(df)}条，期望至少{self.config['min_data_points_per_year']}条"
            )
            report["data_quality_score"] -= 10
        
        # 3. 检查关键指标的异常值
        indicators_to_check = ["pe", "pe_ttm", "pb", "dv_ratio", "dv_ttm"]
        
        for indicator in indicators_to_check:
            if indicator in df.columns:
                outliers = self._detect_outliers(df, indicator)
                if outliers:
                    report["outliers"][indicator] = outliers
                    outlier_count = len(outliers)
                    if outlier_count > len(df) * 0.1:  # 异常值超过10%
                        report["warnings"].append(
                            f"{indicator} 异常值比例过高: {outlier_count}/{len(df)}"
                        )
                        report["data_quality_score"] -= 5
        
        # 4. 检查PE/PB的合理性
        if "pe" in df.columns:
            pe_valid, pe_msg = self._validate_pe(df["pe"])
            if not pe_valid:
                report["warnings"].append(pe_msg)
                report["data_quality_score"] -= 5
        
        if "pb" in df.columns:
            pb_valid, pb_msg = self._validate_pb(df["pb"])
            if not pb_valid:
                report["warnings"].append(pb_msg)
                report["data_quality_score"] -= 5
        
        # 5. 检查日期连续性
        if "trade_date" in df.columns:
            date_gaps = self._check_date_continuity(df["trade_date"])
            if date_gaps:
                report["warnings"].append(f"日期不连续，缺失 {len(date_gaps)} 个交易日")
                report["data_quality_score"] -= 5
        
        # 6. 检查重复数据
        if df.duplicated(subset=["trade_date"]).any():
            duplicate_count = df.duplicated(subset=["trade_date"]).sum()
            report["warnings"].append(f"存在 {duplicate_count} 条重复日期数据")
            report["data_quality_score"] -= 10
        
        # 确保分数不低于0
        report["data_quality_score"] = max(0, report["data_quality_score"])
        
        # 如果分数低于60，标记为无效
        if report["data_quality_score"] < 60:
            report["is_valid"] = False
            report["errors"].append("数据质量分数过低")

        return report

    def validate_yearly_data(self, df: pd.DataFrame,
                            ts_code: str = '',
                            expected_years: int = ANALYSIS_YEARS) -> Dict:
        """
        验证年度财务数据的完整性和准确性

        Args:
            df: 年度数据DataFrame
            ts_code: 股票代码（用于日志记录）
            expected_years: 期望的年份数量

        Returns:
            验证报告字典
        """
        report = {
            "ts_code": ts_code,
            "expected_years": expected_years,
            "actual_years": 0,
            "total_records": len(df),
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "missing_fields": [],
            "missing_years": [],
            "data_quality_score": 100
        }

        if df.empty:
            report["is_valid"] = False
            report["errors"].append("年度数据为空")
            report["data_quality_score"] = 0
            return report

        # 1. 检查必要字段（年度数据使用 end_date 而非 trade_date）
        required_fields = ["ts_code", "end_date"]
        for field in required_fields:
            if field not in df.columns:
                report["missing_fields"].append(field)
                report["warnings"].append(f"缺少字段: {field}")

        # 2. 检查数据量（每年应该有1条年报数据）
        report["actual_years"] = len(df)
        if len(df) < expected_years:
            missing = expected_years - len(df)
            report["warnings"].append(
                f"年份数量不足: 实际{len(df)}年，期望{expected_years}年，缺失{missing}年"
            )
            report["data_quality_score"] -= missing * 15

            # 找出缺失的年份
            if "end_date" in df.columns:
                available_years = set(df["end_date"].str[:4].astype(int))
                current_year = datetime.now().year
                expected_year_set = set(range(current_year - expected_years, current_year))
                report["missing_years"] = sorted(expected_year_set - available_years)

        # 3. 检查估值指标字段（至少要有pe或pb之一）
        has_valuation = any(field in df.columns for field in ["pe", "pe_ttm", "pb"])
        if not has_valuation:
            report["warnings"].append("缺少估值指标字段（pe/pb），可能影响分析结果")
            report["data_quality_score"] -= 10

        # 4. 检查关键指标的异常值
        indicators_to_check = ["roe", "roa", "grossprofit_margin", "debt_to_assets"]

        for indicator in indicators_to_check:
            if indicator in df.columns:
                outliers = self._detect_outliers_yearly(df, indicator)
                if outliers:
                    report["warnings"].append(
                        f"{indicator} 存在异常值: {len(outliers)}条"
                    )
                    report["data_quality_score"] -= 5

        # 5. 检查估值指标合理性
        if "pe" in df.columns or "pe_ttm" in df.columns:
            pe_col = "pe_ttm" if "pe_ttm" in df.columns else "pe"
            pe_valid, pe_msg = self._validate_pe_yearly(df[pe_col])
            if not pe_valid:
                report["warnings"].append(pe_msg)
                report["data_quality_score"] -= 5

        if "pb" in df.columns:
            pb_valid, pb_msg = self._validate_pb_yearly(df["pb"])
            if not pb_valid:
                report["warnings"].append(pb_msg)
                report["data_quality_score"] -= 5

        # 6. 检查重复年份
        if "end_date" in df.columns:
            duplicate_years = df["end_date"].duplicated().sum()
            if duplicate_years > 0:
                report["warnings"].append(f"存在 {duplicate_years} 个重复年份的数据")
                report["data_quality_score"] -= 10

        # 确保分数不低于0
        report["data_quality_score"] = max(0, report["data_quality_score"])

        # 如果分数低于60或年份数量不足要求的一半，标记为无效
        min_required_years = expected_years // 2
        if report["data_quality_score"] < 60 or len(df) < min_required_years:
            report["is_valid"] = False
            if len(df) < min_required_years:
                report["errors"].append(f"年份数量过少，至少需要{min_required_years}年")

        return report

    def _detect_outliers_yearly(self, df: pd.DataFrame,
                                column: str) -> List[Dict]:
        """
        使用IQR方法检测年度数据的异常值

        Args:
            df: DataFrame
            column: 要检查的列名

        Returns:
            异常值列表
        """
        outliers = []
        data = df[column].dropna()

        if len(data) < 3:  # 年度数据至少需要3年才能判断
            return outliers

        # 使用IQR方法
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 2 * IQR
        upper_bound = Q3 + 2 * IQR

        # 针对特定指标设置边界
        if column == "roe":
            upper_bound = min(upper_bound, 100)
            lower_bound = max(lower_bound, -100)
        elif column == "roa":
            upper_bound = min(upper_bound, 50)
            lower_bound = max(lower_bound, -50)
        elif column == "grossprofit_margin":
            upper_bound = min(upper_bound, 100)
            lower_bound = max(lower_bound, 0)
        elif column == "debt_to_assets":
            upper_bound = min(upper_bound, 100)
            lower_bound = max(lower_bound, 0)

        # 找出异常值
        outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        outlier_rows = df[outlier_mask]

        for _, row in outlier_rows.iterrows():
            outliers.append({
                "year": row.get("end_date", "")[:4] if "end_date" in row else "",
                "value": row[column],
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            })

        return outliers

    def _validate_pe_yearly(self, pe_series: pd.Series) -> Tuple[bool, str]:
        """
        验证年度PE数据的合理性

        Returns:
            (是否有效, 消息)
        """
        pe_clean = pe_series.dropna()

        if len(pe_clean) == 0:
            return False, "PE数据全部为空"

        pe_mean = pe_clean.mean()
        pe_max = pe_clean.max()
        pe_min = pe_clean.min()

        # 检查是否有极端值
        if pe_max > self.config["max_pe_threshold"]:
            extreme_count = (pe_clean > self.config["max_pe_threshold"]).sum()
            return False, f"PE存在极端高值: 最大值={pe_max:.2f}，超阈值的数量={extreme_count}"

        if pe_min < 0:
            negative_count = (pe_clean < 0).sum()
            if negative_count > len(pe_clean) * 0.5:
                return False, f"PE负值比例过高: {negative_count}/{len(pe_clean)}"

        return True, "PE数据正常"

    def _validate_pb_yearly(self, pb_series: pd.Series) -> Tuple[bool, str]:
        """
        验证年度PB数据的合理性

        Returns:
            (是否有效, 消息)
        """
        pb_clean = pb_series.dropna()

        if len(pb_clean) == 0:
            return False, "PB数据全部为空"

        pb_max = pb_clean.max()
        pb_min = pb_clean.min()

        if pb_max > self.config["max_pb_threshold"]:
            extreme_count = (pb_clean > self.config["max_pb_threshold"]).sum()
            return False, f"PB存在极端高值: 最大值={pb_max:.2f}，超阈值的数量={extreme_count}"

        if pb_min < 0:
            negative_count = (pb_clean < 0).sum()
            if negative_count > len(pb_clean) * 0.5:
                return False, f"PB负值比例过高: {negative_count}/{len(pb_clean)}"

        return True, "PB数据正常"

    def check_data_completeness(self, data_dict: Dict[str, pd.DataFrame],
                                expected_years: int = ANALYSIS_YEARS) -> Dict:
        """
        检查所有股票的数据完整性

        Args:
            data_dict: 股票代码到数据的映射字典
            expected_years: 期望的年份数量

        Returns:
            完整性报告
        """
        report = {
            "total_stocks": len(data_dict),
            "complete_stocks": 0,
            "incomplete_stocks": 0,
            "missing_stocks": 0,
            "year_distribution": {},
            "incomplete_list": []
        }

        for ts_code, df in data_dict.items():
            if df.empty:
                report["missing_stocks"] += 1
                report["incomplete_list"].append({"ts_code": ts_code, "reason": "无数据"})
                continue

            actual_years = len(df)
            year_count_key = f"{actual_years}年"

            if year_count_key not in report["year_distribution"]:
                report["year_distribution"][year_count_key] = 0
            report["year_distribution"][year_count_key] += 1

            if actual_years >= expected_years:
                report["complete_stocks"] += 1
            else:
                report["incomplete_stocks"] += 1
                report["incomplete_list"].append({
                    "ts_code": ts_code,
                    "reason": f"年份数不足: {actual_years}/{expected_years}"
                })

        return report
    
    def _detect_outliers(self, df: pd.DataFrame, 
                        column: str) -> List[Dict]:
        """
        使用IQR方法检测异常值
        
        Args:
            df: DataFrame
            column: 要检查的列名
            
        Returns:
            异常值列表
        """
        outliers = []
        data = df[column].dropna()
        
        if len(data) < 4:  # 数据太少无法判断
            return outliers
        
        # 使用IQR方法
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - self.config["outlier_std_threshold"] * IQR
        upper_bound = Q3 + self.config["outlier_std_threshold"] * IQR
        
        # 针对特定指标设置边界
        if column in ["pe", "pe_ttm"]:
            upper_bound = min(upper_bound, self.config["max_pe_threshold"])
            lower_bound = max(lower_bound, self.config["min_pe_threshold"])
        elif column in ["pb"]:
            upper_bound = min(upper_bound, self.config["max_pb_threshold"])
            lower_bound = max(lower_bound, self.config["min_pb_threshold"])
        elif column in ["dv_ratio", "dv_ttm"]:
            upper_bound = min(upper_bound, self.config["max_dv_ratio"])
        
        # 找出异常值
        outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        outlier_rows = df[outlier_mask]
        
        for _, row in outlier_rows.iterrows():
            outliers.append({
                "trade_date": row.get("trade_date", ""),
                "value": row[column],
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            })
        
        return outliers
    
    def _validate_pe(self, pe_series: pd.Series) -> Tuple[bool, str]:
        """
        验证PE数据的合理性
        
        Returns:
            (是否有效, 消息)
        """
        pe_clean = pe_series.dropna()
        
        if len(pe_clean) == 0:
            return False, "PE数据全部为空"
        
        pe_mean = pe_clean.mean()
        pe_max = pe_clean.max()
        pe_min = pe_clean.min()
        
        # 检查是否有极端值
        if pe_max > self.config["max_pe_threshold"]:
            extreme_count = (pe_clean > self.config["max_pe_threshold"]).sum()
            return False, f"PE存在极端高值: 最大值={pe_max:.2f}, 超过阈值的数量={extreme_count}"
        
        if pe_min < 0:
            negative_count = (pe_clean < 0).sum()
            if negative_count > len(pe_clean) * 0.5:
                return False, f"PE负值比例过高: {negative_count}/{len(pe_clean)}"
        
        return True, "PE数据正常"
    
    def _validate_pb(self, pb_series: pd.Series) -> Tuple[bool, str]:
        """
        验证PB数据的合理性
        
        Returns:
            (是否有效, 消息)
        """
        pb_clean = pb_series.dropna()
        
        if len(pb_clean) == 0:
            return False, "PB数据全部为空"
        
        pb_mean = pb_clean.mean()
        pb_max = pb_clean.max()
        pb_min = pb_clean.min()
        
        if pb_max > self.config["max_pb_threshold"]:
            extreme_count = (pb_clean > self.config["max_pb_threshold"]).sum()
            return False, f"PB存在极端高值: 最大值={pb_max:.2f}, 超过阈值的数量={extreme_count}"
        
        if pb_min < 0:
            negative_count = (pb_clean < 0).sum()
            if negative_count > len(pb_clean) * 0.5:
                return False, f"PB负值比例过高: {negative_count}/{len(pb_clean)}"
        
        return True, "PB数据正常"
    
    def _check_date_continuity(self, date_series: pd.Series) -> List[str]:
        """
        检查日期连续性
        
        Returns:
            缺失的日期列表
        """
        # 转换为datetime
        dates = pd.to_datetime(date_series).sort_values().reset_index(drop=True)
        
        if len(dates) < 2:
            return []
        
        # 计算日期间隔
        date_diff = dates.diff().dropna()
        
        # 正常交易日间隔应该为1天（周末除外）
        # 这里简化处理，只检查超过5天的间隔
        long_gaps = date_diff[date_diff > pd.Timedelta(days=5)]
        
        missing_dates = []
        for idx, gap in long_gaps.items():
            # idx是gap_end的索引
            if idx > 0:
                start_date = dates.iloc[idx - 1]
                end_date = dates.iloc[idx]
                gap_days = gap.days
                
                missing_dates.append({
                    "start": start_date.strftime('%Y%m%d'),
                    "end": end_date.strftime('%Y%m%d'),
                    "gap_days": gap_days
                })
        
        return missing_dates
    
    def clean_data(self, df: pd.DataFrame, 
                   ts_code: str = '') -> pd.DataFrame:
        """
        清洗数据，处理异常值和缺失值
        
        Args:
            df: 原始数据DataFrame
            ts_code: 股票代码
            
        Returns:
            清洗后的DataFrame
        """
        if df.empty:
            return df
        
        df_clean = df.copy()
        
        # 1. 删除重复数据
        before_count = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=["trade_date"], keep="first")
        after_count = len(df_clean)
        if before_count != after_count:
            logger.info(f"{ts_code}: 删除 {before_count - after_count} 条重复数据")
        
        # 2. 按日期排序
        df_clean = df_clean.sort_values("trade_date").reset_index(drop=True)
        
        # 3. 处理极端异常值（使用边界值替换）
        indicators = ["pe", "pe_ttm", "pb"]
        for indicator in indicators:
            if indicator in df_clean.columns:
                # 获取阈值
                if indicator in ["pe", "pe_ttm"]:
                    upper = self.config["max_pe_threshold"]
                    lower = self.config["min_pe_threshold"]
                elif indicator == "pb":
                    upper = self.config["max_pb_threshold"]
                    lower = self.config["min_pb_threshold"]
                else:
                    continue
                
                # 替换极端值
                extreme_high = df_clean[indicator] > upper
                extreme_low = df_clean[indicator] < lower
                
                if extreme_high.any():
                    count = extreme_high.sum()
                    df_clean.loc[extreme_high, indicator] = np.nan
                    logger.info(f"{ts_code}: {indicator} 标记 {count} 个极端高值为NA")
                
                if extreme_low.any():
                    count = extreme_low.sum()
                    df_clean.loc[extreme_low, indicator] = np.nan
                    logger.info(f"{ts_code}: {indicator} 标记 {count} 个极端低值为NA")
        
        # 4. 处理股息率异常值
        if "dv_ratio" in df_clean.columns:
            extreme_dv = df_clean["dv_ratio"] > self.config["max_dv_ratio"]
            if extreme_dv.any():
                count = extreme_dv.sum()
                df_clean.loc[extreme_dv, "dv_ratio"] = np.nan
                logger.info(f"{ts_code}: dv_ratio 标记 {count} 个极端值为NA")
        
        return df_clean
    
    def validate_stock_list(self, df: pd.DataFrame) -> Dict:
        """
        验证股票列表数据
        
        Args:
            df: 股票列表DataFrame
            
        Returns:
            验证报告
        """
        report = {
            "total_stocks": len(df),
            "is_valid": True,
            "warnings": [],
            "errors": []
        }
        
        if df.empty:
            report["is_valid"] = False
            report["errors"].append("股票列表为空")
            return report
        
        # 检查必要字段
        required_fields = ["ts_code", "name", "list_date"]
        for field in required_fields:
            if field not in df.columns:
                report["errors"].append(f"缺少必要字段: {field}")
                report["is_valid"] = False
        
        # 检查是否有重复代码
        if "ts_code" in df.columns:
            duplicates = df["ts_code"].duplicated().sum()
            if duplicates > 0:
                report["warnings"].append(f"存在 {duplicates} 个重复的股票代码")
        
        # 检查上市日期
        if "list_date" in df.columns:
            invalid_dates = df["list_date"].isna().sum()
            if invalid_dates > 0:
                report["warnings"].append(f"{invalid_dates} 只股票缺少上市日期")
        
        return report
    
    def get_quality_summary(self) -> Dict:
        """获取数据质量汇总报告"""
        return self.quality_report


def validate_analysis_result(result_df: pd.DataFrame) -> Dict:
    """
    验证分析结果的合理性
    
    Args:
        result_df: 分析结果DataFrame
        
    Returns:
        验证报告
    """
    report = {
        "total_stocks": len(result_df),
        "is_valid": True,
        "warnings": [],
        "errors": [],
        "statistics": {}
    }
    
    if result_df.empty:
        report["is_valid"] = False
        report["errors"].append("分析结果为空")
        return report
    
    # 检查关键字段
    key_fields = ["ts_code", "name", "pe_3y_mean", "pb_3y_mean", "dv_ratio_3y_mean"]
    for field in key_fields:
        if field not in result_df.columns:
            report["errors"].append(f"缺少关键字段: {field}")
            report["is_valid"] = False
    
    # 统计各指标的有效数据量
    indicators = ["pe", "pb", "dv_ratio", "dv_ttm"]
    for indicator in indicators:
        mean_col = f"{indicator}_3y_mean"
        if mean_col in result_df.columns:
            valid_count = result_df[mean_col].notna().sum()
            report["statistics"][mean_col] = {
                "valid_count": int(valid_count),
                "valid_rate": valid_count / len(result_df)
            }
    
    # 检查异常值
    if "pe_3y_mean" in result_df.columns:
        high_pe = (result_df["pe_3y_mean"] > 100).sum()
        if high_pe > len(result_df) * 0.1:
            report["warnings"].append(f"{high_pe} 只股票PE均值超过100")
    
    return report


if __name__ == "__main__":
    # 测试代码
    validator = DataValidator()
    
    # 创建测试数据
    test_data = pd.DataFrame({
        "ts_code": ["000001.SZ"] * 10,
        "trade_date": pd.date_range("20240101", periods=10).strftime("%Y%m%d"),
        "pe": [10, 12, 11, 15, 2000, 13, 14, 12, 11, 10],  # 包含异常值
        "pb": [1.5, 1.6, 1.4, 1.7, 1.5, 1.6, 150, 1.5, 1.4, 1.5],  # 包含异常值
        "dv_ratio": [3.0, 3.1, 2.9, 3.2, 3.0, 3.1, 3.0, 60, 2.9, 3.0]  # 包含异常值
    })
    
    print("测试数据:")
    print(test_data)
    
    # 验证数据
    report = validator.validate_daily_basic(test_data, "000001.SZ", 2024)
    print("\n验证报告:")
    print(report)
    
    # 清洗数据
    cleaned = validator.clean_data(test_data, "000001.SZ")
    print("\n清洗后的数据:")
    print(cleaned)
