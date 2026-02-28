"""
筛选模块 V2.0 - 基于原始数据计算评分并筛选

功能：
1. 计算价值评分（基于PE、PB、股息率等估值指标）
2. 计算质量评分（基于ROE、毛利率、现金流等质量指标）
3. 综合评分 = 价值评分 * 0.5 + 质量评分 * 0.5
4. 输出Top 100和Top 200到level1_filtered/目录
"""

import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)




def _parse_complex_to_float(value):
    """将复数字符串或数值转换为 float"""
    if pd.isna(value):
        return None
    if isinstance(value, str):
        # 处理复数字符串格式: (0.044+0j)
        match = re.match(r'\(([\d.]+)\s*([+-])\s*[\d.]+j\)', value)
        if match:
            return float(match.group(1))
        # 尝试直接转换
        try:
            return float(value)
        except ValueError:
            return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

class StockFilterV2:
    """V2.0 股票筛选器 - 计算评分并输出筛选结果"""

    def __init__(self, output_dir: str = None):
        """
        初始化筛选器

        Args:
            output_dir: 筛选结果输出目录
        """
        if output_dir is None:
            # 默认使用项目根目录下的level1_filtered
            project_root = Path(__file__).parent.parent
            output_dir = project_root / "level1_filtered"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"StockFilterV2 初始化完成，输出目录: {self.output_dir}")

    def calculate_value_score(self, row: pd.Series) -> float:
        """
        计算价值评分（0-100分）

        价值评分基于：
        1. PE估值（越低越好）
        2. PB估值（越低越好）
        3. 股息率（越高越好）
        4. PS估值（越低越好）

        Args:
            row: 股票数据行

        Returns:
            价值评分 (0-100)
        """
        score = 0.0

        # 1. PE评分 (0-30分)
        pe = row.get('pe_ttm') or row.get('pe')
        if pd.notna(pe) and pe > 0:
            if pe < 8:
                score += 30
            elif pe < 12:
                score += 25
            elif pe < 15:
                score += 20
            elif pe < 20:
                score += 15
            elif pe < 30:
                score += 10
            elif pe < 50:
                score += 5
            # PE > 50 得0分

        # 2. PB评分 (0-25分)
        pb = row.get('pb')
        if pd.notna(pb) and pb > 0:
            if pb < 1:
                score += 25
            elif pb < 1.5:
                score += 20
            elif pb < 2:
                score += 15
            elif pb < 3:
                score += 10
            elif pb < 5:
                score += 5
            # PB > 5 得0分

        # 3. 股息率评分 (0-25分)
        dv = row.get('dv_ttm') or row.get('dv_ratio')
        if pd.notna(dv) and dv > 0:
            if dv >= 5:
                score += 25
            elif dv >= 4:
                score += 20
            elif dv >= 3:
                score += 15
            elif dv >= 2:
                score += 10
            elif dv >= 1:
                score += 5
            # 股息率 < 1% 得0分

        # 4. PS评分 (0-20分)
        ps = row.get('ps_ttm') or row.get('ps')
        if pd.notna(ps) and ps > 0:
            if ps < 1:
                score += 20
            elif ps < 2:
                score += 15
            elif ps < 3:
                score += 10
            elif ps < 5:
                score += 5
            # PS > 5 得0分

        return min(score, 100)

    def calculate_quality_score(self, row: pd.Series) -> float:
        """
        计算质量评分（0-100分）

        质量评分基于：
        1. ROE水平（越高越好）
        2. ROE稳定性
        3. 毛利率水平（越高越好）
        4. 负债率/流动比率（财务健康度）
        5. 经营现金流
        6. 增长率（EPS和收入增长）

        Args:
            row: 股票数据行

        Returns:
            质量评分 (0-100)
        """
        score = 0.0

        # 1. ROE评分 (0-30分)
        roe = row.get('roe') or row.get('roe_waa') or row.get('roe_dt')
        if pd.notna(roe):
            if roe >= 20:
                score += 30
            elif roe >= 15:
                score += 25
            elif roe >= 12:
                score += 20
            elif roe >= 10:
                score += 15
            elif roe >= 8:
                score += 10
            elif roe >= 5:
                score += 5
            # ROE < 5% 得0分

        # 2. 毛利率评分 (0-25分)
        gross_margin = row.get('gross_margin')
        if pd.notna(gross_margin):
            if gross_margin >= 50:
                score += 25
            elif gross_margin >= 40:
                score += 20
            elif gross_margin >= 30:
                score += 15
            elif gross_margin >= 20:
                score += 10
            elif gross_margin >= 10:
                score += 5
            # 毛利率 < 10% 得0分

        # 3. 流动比率评分 (0-15分)
        current_ratio = row.get('current_ratio')
        if pd.notna(current_ratio):
            if current_ratio >= 2:
                score += 15
            elif current_ratio >= 1.5:
                score += 12
            elif current_ratio >= 1.2:
                score += 8
            elif current_ratio >= 1:
                score += 5
            # 流动比率 < 1 得0分

        # 4. 现金流评分 (0-15分)
        # 使用自由现金流作为指标
        fcfe = row.get('fcfe')
        if pd.notna(fcfe) and fcfe > 0:
            score += 15
        elif pd.notna(fcfe) and fcfe > -1:
            score += 8
        # 负现金流得0分

        # 5. EPS增长评分 (0-15分)
        eps_cagr = row.get('eps_5y_cagr')
        if pd.notna(eps_cagr):
            try:
                # 处理复数字符串格式: (0.044+0j)
                if isinstance(eps_cagr, str):
                    import re
                    match = re.match(r'\(([\d.]+)\s*([+-])\s*[\d.]+j\)', eps_cagr)
                    if match:
                        eps_cagr = float(match.group(1))
                    else:
                        eps_cagr = float(eps_cagr)
                else:
                    eps_cagr = float(eps_cagr)
                
                if eps_cagr >= 0.20:  # 20%以上
                    score += 15
                elif eps_cagr >= 0.15:  # 15%以上
                    score += 12
                elif eps_cagr >= 0.10:  # 10%以上
                    score += 10
                elif eps_cagr >= 0.05:  # 5%以上
                    score += 5
                # 增长率 < 5% 或负增长得0分
            except (ValueError, TypeError):
                pass  # 无法转换则跳过

        return min(score, 100)

    def calculate_total_score(self, value_score: float, quality_score: float,
                             value_weight: float = 0.5, quality_weight: float = 0.5) -> float:
        """
        计算综合评分

        Args:
            value_score: 价值评分
            quality_score: 质量评分
            value_weight: 价值权重
            quality_weight: 质量权重

        Returns:
            综合评分 (0-100)
        """
        return value_score * value_weight + quality_score * quality_weight

    def get_rating(self, total_score: float) -> str:
        """
        根据综合评分获取评级

        Args:
            total_score: 综合评分

        Returns:
            评级字符串
        """
        if total_score >= 85:
            return "A+"
        elif total_score >= 75:
            return "A"
        elif total_score >= 65:
            return "B+"
        elif total_score >= 55:
            return "B"
        elif total_score >= 45:
            return "C+"
        elif total_score >= 35:
            return "C"
        else:
            return "D"

    def filter_and_score(self, df: pd.DataFrame,
                        min_data_years: int = 3,
                        exclude_st: bool = True,
                        exclude_financials: bool = False) -> pd.DataFrame:
        """
        对原始数据进行筛选和评分

        Args:
            df: 原始数据DataFrame
            min_data_years: 最少数据年限
            exclude_st: 是否排除ST股票
            exclude_financials: 是否排除金融股

        Returns:
            评分后的DataFrame
        """
        if df.empty:
            logger.error("输入数据为空")
            return pd.DataFrame()

        logger.info(f"开始筛选和评分，原始数据: {len(df)} 只股票")

        # 复制数据
        result_df = df.copy()

        # 1. 数据年限筛选
        if 'data_years' in result_df.columns:
            result_df = result_df[result_df['data_years'] >= min_data_years]
            logger.info(f"数据年限筛选后: {len(result_df)} 只股票")

        # 2. 排除ST股票
        if exclude_st and 'name' in result_df.columns:
            result_df = result_df[~result_df['name'].str.contains('ST', na=False)]
            logger.info(f"排除ST后: {len(result_df)} 只股票")

        # 3. 排除金融股（可选）
        if exclude_financials and 'industry' in result_df.columns:
            financial_keywords = ['银行', '保险', '证券', '信托', '租赁', '金融']
            mask = ~result_df['industry'].str.contains('|'.join(financial_keywords), na=False)
            result_df = result_df[mask]
            logger.info(f"排除金融股后: {len(result_df)} 只股票")

        # 4. 计算各项评分
        logger.info("计算评分...")
        result_df['value_score'] = result_df.apply(self.calculate_value_score, axis=1)
        result_df['quality_score'] = result_df.apply(self.calculate_quality_score, axis=1)
        result_df['total_score'] = result_df.apply(
            lambda row: self.calculate_total_score(row['value_score'], row['quality_score']),
            axis=1
        )
        result_df['rating'] = result_df['total_score'].apply(self.get_rating)

        # 5. 排序
        result_df = result_df.sort_values('total_score', ascending=False).reset_index(drop=True)
        result_df['rank'] = range(1, len(result_df) + 1)

        logger.info(f"评分完成，有效股票: {len(result_df)} 只")
        return result_df

    def export_top_n(self, df: pd.DataFrame, top_n: int, filename: str = None) -> str:
        """
        导出Top N股票到CSV文件

        Args:
            df: 评分后的DataFrame
            top_n: 前N名
            filename: 文件名（不含路径和扩展名）

        Returns:
            导出的文件路径
        """
        if df.empty:
            logger.error("数据为空，无法导出")
            return ""

        if filename is None:
            filename = f"top_{top_n}_{datetime.now().strftime('%Y%m%d')}"

        # 取前N名
        top_df = df.head(top_n).copy()

        # 选择要导出的列
        export_columns = [
            'rank', 'ts_code', 'symbol', 'name', 'industry', 'area',
            'value_score', 'quality_score', 'total_score', 'rating',
            'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio', 'dv_ttm',
            'roe', 'roe_waa', 'gross_margin', 'current_ratio', 'quick_ratio',
            'eps_5y_cagr', 'revenue_5y_cagr', 'data_years',
            'total_mv', 'circ_mv', 'close', 'list_date'
        ]

        # 只保留存在的列
        export_columns = [col for col in export_columns if col in top_df.columns]
        export_df = top_df[export_columns]

        filepath = self.output_dir / f"{filename}.csv"
        export_df.to_csv(filepath, index=False, encoding='utf-8-sig')

        logger.info(f"Top {top_n} 已导出: {filepath}")
        return str(filepath)

    def export_summary_stats(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        导出汇总统计信息

        Args:
            df: 评分后的DataFrame
            filename: 文件名

        Returns:
            导出的文件路径
        """
        if df.empty:
            return ""

        if filename is None:
            filename = f"summary_stats_{datetime.now().strftime('%Y%m%d')}"

        # 计算统计信息
        stats = {
            '统计项目': [],
            '数值': []
        }

        stats['统计项目'].extend([
            '总股票数', '平均价值评分', '平均质量评分', '平均综合评分',
            'A+级数量', 'A级数量', 'B+级数量', 'B级数量', 'C级数量', 'D级数量'
        ])

        stats['数值'].extend([
            len(df),
            f"{df['value_score'].mean():.2f}",
            f"{df['quality_score'].mean():.2f}",
            f"{df['total_score'].mean():.2f}",
            len(df[df['rating'] == 'A+']),
            len(df[df['rating'] == 'A']),
            len(df[df['rating'] == 'B+']),
            len(df[df['rating'] == 'B']),
            len(df[df['rating'].str.contains('C', na=False)]),
            len(df[df['rating'] == 'D'])
        ])

        # 行业分布（Top 10）
        if 'industry' in df.columns:
            industry_counts = df['industry'].value_counts().head(10)
            stats['统计项目'].append('Top 1 行业')
            stats['数值'].append(f"{industry_counts.index[0]} ({industry_counts.iloc[0]}只)")

        stats_df = pd.DataFrame(stats)
        filepath = self.output_dir / f"{filename}.csv"
        stats_df.to_csv(filepath, index=False, encoding='utf-8-sig')

        logger.info(f"汇总统计已导出: {filepath}")
        return str(filepath)

    def process_and_export(self, df: pd.DataFrame,
                          top_list: List[int] = [100, 200],
                          export_summary: bool = True,
                          **filter_kwargs) -> Dict[str, str]:
        """
        完整处理流程：筛选、评分、导出

        Args:
            df: 原始数据DataFrame
            top_list: 要导出的Top N列表，如[100, 200]
            export_summary: 是否导出汇总统计
            **filter_kwargs: 传递给filter_and_score的额外参数

        Returns:
            导出文件路径字典
        """
        # 筛选和评分
        scored_df = self.filter_and_score(df, **filter_kwargs)

        if scored_df.empty:
            logger.error("筛选后无数据")
            return {}

        # 导出结果
        exported_files = {}

        for top_n in top_list:
            filepath = self.export_top_n(scored_df, top_n)
            exported_files[f'top_{top_n}'] = filepath

        # 导出汇总
        if export_summary:
            summary_path = self.export_summary_stats(scored_df)
            exported_files['summary'] = summary_path

        # 导出完整评分数据
        full_filename = f"full_scored_{datetime.now().strftime('%Y%m%d')}"
        full_path = self.output_dir / f"{full_filename}.csv"
        scored_df.to_csv(full_path, index=False, encoding='utf-8-sig')
        exported_files['full'] = str(full_path)

        logger.info(f"导出完成，共 {len(exported_files)} 个文件")

        return exported_files


if __name__ == "__main__":
    # 测试代码
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    try:
        # 创建测试数据
        test_data = {
            'ts_code': ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ'],
            'symbol': ['000001', '000002', '600000', '600036', '000858'],
            'name': ['平安银行', '万科A', '浦发银行', '招商银行', '五粮液'],
            'industry': ['银行', '房地产', '银行', '银行', '酿酒行业'],
            'area': ['深圳', '深圳', '上海', '深圳', '四川'],
            'list_date': ['19910403', '19910129', '19991110', '20020409', '19980427'],
            'pe': [5.5, 8.2, 4.8, 6.1, 25.3],
            'pe_ttm': [5.8, 9.1, 5.2, 6.5, 28.5],
            'pb': [0.7, 1.2, 0.5, 1.1, 8.5],
            'ps': [2.1, 1.5, 1.8, 4.2, 12.3],
            'dv_ratio': [3.5, 4.2, 4.8, 3.8, 2.5],
            'roe': [12.5, 15.3, 10.8, 16.2, 25.5],
            'roe_waa': [12.8, 15.8, 11.2, 16.8, 26.1],
            'gross_margin': [45.2, 28.5, 42.1, 48.5, 75.2],
            'current_ratio': [1.2, 1.8, 1.1, 1.3, 2.5],
            'quick_ratio': [1.1, 0.5, 1.0, 1.2, 1.8],
            'eps_5y_cagr': [0.08, 0.05, 0.03, 0.12, 0.18],
            'revenue_5y_cagr': [0.10, 0.08, 0.05, 0.15, 0.12],
            'data_years': [5, 5, 5, 5, 5],
            'fcfe': [100000, 50000, 80000, 150000, 200000],
            'total_mv': [25000000, 30000000, 18000000, 120000000, 320000000],
            'close': [12.5, 18.2, 8.5, 35.6, 185.2],
        }

        test_df = pd.DataFrame(test_data)

        # 测试筛选器
        filter_v2 = StockFilterV2()
        print("测试筛选和评分...")

        # 计算评分
        scored_df = filter_v2.filter_and_score(test_df)

        print("\n评分结果:")
        print(scored_df[['name', 'value_score', 'quality_score', 'total_score', 'rating']])

        # 测试导出
        print("\n测试导出...")
        exported = filter_v2.process_and_export(test_df, top_list=[5])
        print(f"导出文件: {exported}")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
