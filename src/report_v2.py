"""
报告模块 V2.0 - 生成Value Line风格的股票报告

功能：
1. 为Top 100股票生成详细的Value Line风格报告
2. 报告包含：基础信息、核心指标、增长趋势、价值线图表占位
3. 输出到level2_reports/目录，每只股票一个独立文件
"""

import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import json

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


class ValueLineReportV2:
    """V2.0 Value Line风格报告生成器"""

    def __init__(self, output_dir: str = None):
        """
        初始化报告生成器

        Args:
            output_dir: 报告输出目录
        """
        if output_dir is None:
            # 默认使用项目根目录下的level2_reports
            project_root = Path(__file__).parent.parent
            output_dir = project_root / "level2_reports"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ValueLineReportV2 初始化完成，输出目录: {self.output_dir}")

    def generate_stock_report(self, stock_data: Dict) -> str:
        """
        为单只股票生成Value Line风格报告

        Args:
            stock_data: 股票数据字典

        Returns:
            报告文件路径
        """
        ts_code = stock_data.get('ts_code', 'UNKNOWN')
        name = stock_data.get('name', '未知')

        # 生成报告内容
        report_lines = self._build_report_content(stock_data)

        # 保存报告
        filename = f"{ts_code}_{name}_{datetime.now().strftime('%Y%m%d')}.txt"
        filepath = self.output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        return str(filepath)

    def _build_report_content(self, stock_data: Dict) -> List[str]:
        """
        构建报告内容

        Args:
            stock_data: 股票数据字典

        Returns:
            报告行列表
        """
        lines = []

        # ==================== 报告头部 ====================
        lines.append("=" * 80)
        lines.append(f"VALUE LINE 风格股票分析报告")
        lines.append("=" * 80)
        lines.append("")

        # ==================== 基础信息 ====================
        lines.append("-" * 80)
        lines.append("【基础信息】")
        lines.append("-" * 80)
        lines.append(f"股票代码:    {stock_data.get('ts_code', 'N/A')}")
        lines.append(f"股票名称:    {stock_data.get('name', 'N/A')}")
        lines.append(f"所属行业:    {stock_data.get('industry', 'N/A')}")
        lines.append(f"所在地域:    {stock_data.get('area', 'N/A')}")
        lines.append(f"市场类型:    {stock_data.get('market', 'N/A')}")
        lines.append(f"交易所:      {stock_data.get('exchange', 'N/A')}")
        lines.append(f"上市日期:    {stock_data.get('list_date', 'N/A')}")
        lines.append(f"是否沪深港通: {stock_data.get('is_hs', 'N/A')}")
        lines.append("")

        # ==================== 综合评级 ====================
        lines.append("-" * 80)
        lines.append("【综合评级】")
        lines.append("-" * 80)

        total_score = stock_data.get('total_score', 0)
        rating = stock_data.get('rating', 'N/A')
        value_score = stock_data.get('value_score', 0)
        quality_score = stock_data.get('quality_score', 0)
        rank = stock_data.get('rank', 'N/A')

        lines.append(f"综合排名:    #{rank}")
        lines.append(f"综合评分:    {total_score:.1f} / 100")
        lines.append(f"投资评级:    {rating}")
        lines.append(f"  - 价值评分: {value_score:.1f} / 100")
        lines.append(f"  - 质量评分: {quality_score:.1f} / 100")
        lines.append("")

        # ==================== 估值指标 ====================
        lines.append("-" * 80)
        lines.append("【估值指标】")
        lines.append("-" * 80)

        pe = stock_data.get('pe_ttm') or stock_data.get('pe')
        pb = stock_data.get('pb')
        ps = stock_data.get('ps_ttm') or stock_data.get('ps')
        dv = stock_data.get('dv_ttm') or stock_data.get('dv_ratio')
        close = stock_data.get('close')
        total_mv = stock_data.get('total_mv')
        circ_mv = stock_data.get('circ_mv')

        lines.append(f"最新价格:    {close:.2f} 元" if pd.notna(close) else "最新价格:    N/A")
        lines.append(f"市盈率(TTM): {pe:.2f} 倍" if pd.notna(pe) else "市盈率(TTM): N/A")
        lines.append(f"市净率:      {pb:.2f} 倍" if pd.notna(pb) else "市净率:      N/A")
        lines.append(f"市销率(TTM): {ps:.2f} 倍" if pd.notna(ps) else "市销率(TTM): N/A")
        lines.append(f"股息率(TTM): {dv:.2f} %" if pd.notna(dv) else "股息率(TTM): N/A")
        lines.append(f"总市值:      {total_mv/10000:.2f} 亿元" if pd.notna(total_mv) else "总市值:      N/A")
        lines.append(f"流通市值:    {circ_mv/10000:.2f} 亿元" if pd.notna(circ_mv) else "流通市值:    N/A")
        lines.append("")

        # ==================== 盈利能力指标 ====================
        lines.append("-" * 80)
        lines.append("【盈利能力】")
        lines.append("-" * 80)

        roe = stock_data.get('roe') or stock_data.get('roe_waa')
        roe_dt = stock_data.get('roe_dt')
        gross_margin = stock_data.get('gross_margin')

        lines.append(f"净资产收益率(ROE): {roe:.2f} %" if pd.notna(roe) else "净资产收益率(ROE): N/A")
        lines.append(f"扣非ROE:          {roe_dt:.2f} %" if pd.notna(roe_dt) else "扣非ROE:          N/A")
        lines.append(f"毛利率:            {gross_margin:.2f} %" if pd.notna(gross_margin) else "毛利率:            N/A")
        lines.append("")

        # ==================== 财务健康度 ====================
        lines.append("-" * 80)
        lines.append("【财务健康度】")
        lines.append("-" * 80)

        current_ratio = stock_data.get('current_ratio')
        quick_ratio = stock_data.get('quick_ratio')
        cash_ratio = stock_data.get('cash_ratio')
        fcfe = stock_data.get('fcfe')
        fcff = stock_data.get('fcff')

        lines.append(f"流动比率:     {current_ratio:.2f}" if pd.notna(current_ratio) else "流动比率:     N/A")
        lines.append(f"速动比率:     {quick_ratio:.2f}" if pd.notna(quick_ratio) else "速动比率:     N/A")
        lines.append(f"现金比率:     {cash_ratio:.2f}" if pd.notna(cash_ratio) else "现金比率:     N/A")
        lines.append(f"股权自由现金流: {fcfe/100000000:.2f} 亿元" if pd.notna(fcfe) else "股权自由现金流: N/A")
        lines.append(f"企业自由现金流: {fcff/100000000:.2f} 亿元" if pd.notna(fcff) else "企业自由现金流: N/A")
        lines.append("")

        # ==================== 增长指标 ====================
        lines.append("-" * 80)
        lines.append("【增长指标(5年复合)】")
        lines.append("-" * 80)

        eps_cagr = stock_data.get('eps_5y_cagr')
        revenue_cagr = stock_data.get('revenue_5y_cagr')
        data_years = stock_data.get('data_years', 0)

        if pd.notna(eps_cagr):
            lines.append(f"每股收益增长: {eps_cagr*100:.2f} % (CAGR)")
        else:
            lines.append("每股收益增长: N/A")

        if pd.notna(revenue_cagr):
            lines.append(f"营业收入增长: {revenue_cagr*100:.2f} % (CAGR)")
        else:
            lines.append("营业收入增长: N/A")

        lines.append(f"数据年限:     {data_years} 年")
        lines.append("")

        # ==================== 估值评估 ====================
        lines.append("-" * 80)
        lines.append("【估值评估】")
        lines.append("-" * 80)

        # 根据PE和PB给出估值判断
        valuation_comment = self._get_valuation_comment(stock_data)
        for comment in valuation_comment:
            lines.append(comment)
        lines.append("")

        # ==================== 投资建议 ====================
        lines.append("-" * 80)
        lines.append("【投资建议】")
        lines.append("-" * 80)

        recommendation = self._get_recommendation(stock_data)
        for line in recommendation:
            lines.append(line)
        lines.append("")

        # ==================== Value Line 图表占位 ====================
        lines.append("-" * 80)
        lines.append("【Value Line 图表占位】")
        lines.append("-" * 80)
        lines.append("")
        lines.append("┌────────────────────────────────────────────────────────────────┐")
        lines.append("│  Price                                                          │")
        lines.append("│    ↑                                                           │")
        lines.append("│  $ │                                                            │")
        lines.append("│    │                                                            │")
        lines.append("│    │                (价格趋势图占位)                            │")
        lines.append("│    │            *  (数据点将在后续版本中绘制)                   │")
        lines.append("│    │          * * *                                              │")
        lines.append("│    │        * * * * *                                            │")
        lines.append("│  $ │──────────────────────────────────────────────────→       │")
        lines.append("│    0     2     4     6     8     10    (Years)                  │")
        lines.append("└────────────────────────────────────────────────────────────────┘")
        lines.append("")
        lines.append("图例:")
        lines.append("  - 实线: 历史价格")
        lines.append("  - 虚线: 内在价值估算")
        lines.append("  - 阴影: 安全边际区域")
        lines.append("")

        # ==================== 风险提示 ====================
        lines.append("-" * 80)
        lines.append("【风险提示】")
        lines.append("-" * 80)
        lines.append("1. 本报告基于历史财务数据分析，不构成投资建议")
        lines.append("2. 股票投资有风险，请结合公司基本面、行业趋势、宏观经济综合判断")
        lines.append("3. 建议关注公司定期报告和最新公告")
        lines.append("4. 估值指标仅供参考，实际投资决策需考虑更多因素")
        lines.append("")

        # ==================== 报告尾部 ====================
        lines.append("=" * 80)
        lines.append(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("数据来源: Tushare Pro / A股基本面分析器 V2.0")
        lines.append("=" * 80)

        return lines

    def _get_valuation_comment(self, stock_data: Dict) -> List[str]:
        """生成估值评估评论"""
        comments = []

        pe = stock_data.get('pe_ttm') or stock_data.get('pe')
        pb = stock_data.get('pb')
        dv = stock_data.get('dv_ttm') or stock_data.get('dv_ratio')

        # PE评估
        if pd.notna(pe):
            if pe < 10:
                comments.append(f"  • 市盈率({pe:.1f})处于较低水平，显示估值吸引力")
            elif pe < 20:
                comments.append(f"  • 市盈率({pe:.1f})处于合理区间")
            elif pe < 30:
                comments.append(f"  • 市盈率({pe:.1f})偏高，需关注盈利增长")
            else:
                comments.append(f"  • 市盈率({pe:.1f})较高，估值偏贵")
        else:
            comments.append("  • 市盈率数据缺失")

        # PB评估
        if pd.notna(pb):
            if pb < 1:
                comments.append(f"  • 市净率({pb:.2f})低于1，可能存在破净机会")
            elif pb < 2:
                comments.append(f"  • 市净率({pb:.2f})处于合理水平")
            elif pb < 3:
                comments.append(f"  • 市净率({pb:.2f})适中")
            else:
                comments.append(f"  • 市净率({pb:.2f})较高，市场预期较强")

        # 股息率评估
        if pd.notna(dv):
            if dv >= 4:
                comments.append(f"  • 股息率({dv:.2f}%)较高，具有现金回报吸引力")
            elif dv >= 2:
                comments.append(f"  • 股息率({dv:.2f}%)适中")
            else:
                comments.append(f"  • 股息率({dv:.2f}%)较低，公司偏向成长策略")

        return comments

    def _get_recommendation(self, stock_data: Dict) -> List[str]:
        """生成投资建议"""
        rating = stock_data.get('rating', 'D')
        total_score = stock_data.get('total_score', 0)
        value_score = stock_data.get('value_score', 0)
        quality_score = stock_data.get('quality_score', 0)

        recommendations = []

        if rating in ['A+', 'A']:
            recommendations.append(f"  综合评级: {rating} (强烈推荐)")
            recommendations.append("")
            recommendations.append("  该股票在估值和盈利能力方面均表现优秀，具有较好的投资价值。")
            recommendations.append("  建议关注回调机会，可作为长期投资标的。")
        elif rating in ['B+', 'B']:
            recommendations.append(f"  综合评级: {rating} (推荐)")
            recommendations.append("")
            if value_score > quality_score:
                recommendations.append("  该股票估值优势明显，但盈利能力有待提升。")
                recommendations.append("  适合价值投资者，建议关注盈利改善情况。")
            else:
                recommendations.append("  该股票盈利能力较强，但估值偏高。")
                recommendations.append("  建议等待更好的买入时机。")
        else:
            recommendations.append(f"  综合评级: {rating} (谨慎/回避)")
            recommendations.append("")
            recommendations.append("  该股票当前投资价值有限，建议谨慎对待。")
            recommendations.append("  如需投资，建议深入研究公司基本面和行业前景。")

        return recommendations

    def generate_all_reports(self, top_stocks_df: pd.DataFrame, limit: int = 100) -> List[str]:
        """
        为Top N股票批量生成报告

        Args:
            top_stocks_df: Top股票DataFrame
            limit: 生成报告的数量限制

        Returns:
            生成的报告文件路径列表
        """
        if top_stocks_df.empty:
            logger.error("输入数据为空")
            return []

        limit = min(limit, len(top_stocks_df))
        logger.info(f"开始为 Top {limit} 股票生成报告...")

        report_files = []

        for idx in range(limit):
            stock_data = top_stocks_df.iloc[idx].to_dict()

            try:
                filepath = self.generate_stock_report(stock_data)
                report_files.append(filepath)

                if (idx + 1) % 10 == 0:
                    logger.info(f"已完成 {idx + 1}/{limit} 份报告")

            except Exception as e:
                ts_code = stock_data.get('ts_code', 'UNKNOWN')
                logger.error(f"生成 {ts_code} 报告失败: {e}")

        logger.info(f"报告生成完成，共 {len(report_files)} 份")
        return report_files

    def generate_summary_index(self, report_files: List[str],
                               top_stocks_df: pd.DataFrame) -> str:
        """
        生成报告汇总索引文件

        Args:
            report_files: 报告文件路径列表
            top_stocks_df: Top股票DataFrame

        Returns:
            索引文件路径
        """
        lines = []
        lines.append("=" * 80)
        lines.append("Top 100 股票 Value Line 报告索引")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"报告数量: {len(report_files)}")
        lines.append("")
        lines.append("-" * 80)
        lines.append("")

        # 按排名列出所有股票
        for idx, row in top_stocks_df.head(len(report_files)).iterrows():
            rank = row.get('rank', idx + 1)
            ts_code = row.get('ts_code', 'N/A')
            name = row.get('name', 'N/A')
            rating = row.get('rating', 'N/A')
            total_score = row.get('total_score', 0)

            lines.append(f"#{rank:3d}  {ts_code}  {name:8s}  {rating}  ({total_score:.1f}分)")

        lines.append("")
        lines.append("=" * 80)

        # 保存索引文件
        index_path = self.output_dir / f"index_{datetime.now().strftime('%Y%m%d')}.txt"
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        logger.info(f"索引文件已生成: {index_path}")
        return str(index_path)

    def export_json_format(self, top_stocks_df: pd.DataFrame,
                           limit: int = 100) -> str:
        """
        导出JSON格式的数据（便于其他程序使用）

        Args:
            top_stocks_df: Top股票DataFrame
            limit: 导出数量限制

        Returns:
            JSON文件路径
        """
        limit = min(limit, len(top_stocks_df))

        # 选择要导出的列
        export_data = []
        for idx in range(limit):
            row = top_stocks_df.iloc[idx]

            stock_json = {
                'rank': int(row.get('rank', idx + 1)),
                'ts_code': row.get('ts_code', ''),
                'name': row.get('name', ''),
                'industry': row.get('industry', ''),
                'rating': row.get('rating', ''),
                'total_score': float(row.get('total_score', 0)),
                'value_score': float(row.get('value_score', 0)),
                'quality_score': float(row.get('quality_score', 0)),
                'valuation': {
                    'pe': float(row['pe']) if pd.notna(row.get('pe')) else None,
                    'pe_ttm': float(row['pe_ttm']) if pd.notna(row.get('pe_ttm')) else None,
                    'pb': float(row['pb']) if pd.notna(row.get('pb')) else None,
                    'ps': float(row['ps']) if pd.notna(row.get('ps')) else None,
                    'dv_ratio': float(row['dv_ratio']) if pd.notna(row.get('dv_ratio')) else None,
                },
                'profitability': {
                    'roe': float(row['roe']) if pd.notna(row.get('roe')) else None,
                    'gross_margin': float(row['gross_margin']) if pd.notna(row.get('gross_margin')) else None,
                },
                'growth': {
                    'eps_5y_cagr': _parse_complex_to_float(row.get('eps_5y_cagr')),
                    'revenue_5y_cagr': _parse_complex_to_float(row.get('revenue_5y_cagr')),
                }
            }
            export_data.append(stock_json)

        # 保存JSON文件
        json_path = self.output_dir / f"top_{limit}_{datetime.now().strftime('%Y%m%d')}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        logger.info(f"JSON文件已导出: {json_path}")
        return str(json_path)


if __name__ == "__main__":
    # 测试代码
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    try:
        # 创建测试数据
        test_data = {
            'rank': [1, 2, 3],
            'ts_code': ['000001.SZ', '000002.SZ', '600036.SH'],
            'name': ['平安银行', '万科A', '招商银行'],
            'industry': ['银行', '房地产', '银行'],
            'area': ['深圳', '深圳', '深圳'],
            'market': ['主板', '主板', '主板'],
            'exchange': ['SZSE', 'SZSE', 'SSE'],
            'list_date': ['19910403', '19910129', '20020409'],
            'is_hs': ['S', 'S', 'S'],
            'total_score': [85.5, 78.2, 82.1],
            'value_score': [82.0, 75.5, 80.0],
            'quality_score': [89.0, 80.9, 84.2],
            'rating': ['A+', 'A', 'A'],
            'pe': [5.5, 8.2, 6.1],
            'pe_ttm': [5.8, 9.1, 6.5],
            'pb': [0.7, 1.2, 1.1],
            'ps': [2.1, 1.5, 4.2],
            'dv_ratio': [3.5, 4.2, 3.8],
            'roe': [12.5, 15.3, 16.2],
            'gross_margin': [45.2, 28.5, 48.5],
            'current_ratio': [1.2, 1.8, 1.3],
            'eps_5y_cagr': [0.08, 0.05, 0.12],
            'revenue_5y_cagr': [0.10, 0.08, 0.15],
            'data_years': [5, 5, 5],
            'fcfe': [100000, 50000, 150000],
            'fcff': [120000, 60000, 180000],
            'total_mv': [25000000, 30000000, 120000000],
            'circ_mv': [20000000, 25000000, 100000000],
            'close': [12.5, 18.2, 35.6],
        }

        test_df = pd.DataFrame(test_data)

        # 测试报告生成器
        reporter = ValueLineReportV2()
        print("测试报告生成...")

        # 生成单只股票报告
        stock_data = test_df.iloc[0].to_dict()
        report_path = reporter.generate_stock_report(stock_data)
        print(f"单份报告: {report_path}")

        # 批量生成报告
        print("\n批量生成报告...")
        report_files = reporter.generate_all_reports(test_df, limit=3)
        print(f"生成报告数: {len(report_files)}")

        # 生成索引
        index_path = reporter.generate_summary_index(report_files, test_df)
        print(f"索引文件: {index_path}")

        # 导出JSON
        json_path = reporter.export_json_format(test_df, limit=3)
        print(f"JSON文件: {json_path}")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
