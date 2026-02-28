"""
主程序 V2.0 - A股基本面分析器分级输出版本

功能：
1. 串联整个分析流程：数据获取 -> 筛选评分 -> 报告生成
2. 支持命令行参数控制执行范围
3. 实现分级输出：raw_data -> level1_filtered -> level2_reports

使用方法：
    python main_v2.py --full              # 完整流程
    python main_v2.py --fetch-only        # 仅获取数据
    python main_v2.py --filter-only       # 仅筛选评分
    python main_v2.py --report-only       # 仅生成报告
    python main_v2.py --fetch --filter    # 获取数据+筛选评分

配置：
    在config.py中设置TUSHARE_TOKEN
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# 导入V2模块
from fetcher_v2 import DataFetcherV2
from filter_v2 import StockFilterV2
from report_v2 import ValueLineReportV2

# 导入配置
from config import TUSHARE_TOKEN

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            f"logs/main_v2_{datetime.now().strftime('%Y%m%d')}.log",
            encoding='utf-8'
        )
    ]
)
logger = logging.getLogger(__name__)


class StockAnalyzerV2:
    """V2.0 A股分析器主类 - 支持分级输出"""

    def __init__(self, token: str = None):
        """
        初始化分析器

        Args:
            token: Tushare Pro API Token
        """
        self.token = token or TUSHARE_TOKEN
        if not self.token or self.token == "your_token_here":
            raise ValueError(
                "请提供有效的Tushare Pro API Token！\n"
                "1. 访问 https://tushare.pro/register 注册账号\n"
                "2. 获取Token后在config.py中设置，或通过环境变量TUSHARE_TOKEN传入"
            )

        # 初始化各模块
        self.fetcher = DataFetcherV2(token=self.token)
        self.filter = StockFilterV2()
        self.reporter = ValueLineReportV2()

        # 数据存储
        self.raw_data = None
        self.filtered_data = None

        logger.info("StockAnalyzerV2 初始化完成")

    def run_fetch(self, years: int = 5, sample: int = None) -> str:
        """
        执行数据获取流程

        Args:
            years: 财务数据年限
            sample: 只获取前N只股票（用于测试）

        Returns:
            原始数据文件路径
        """
        logger.info("=" * 60)
        logger.info("开始执行数据获取流程")
        logger.info("=" * 60)

        start_time = time.time()

        # 获取并保存数据
        filepath = self.fetcher.fetch_and_save(
            years=years,
            sample=sample,
            progress_callback=self._progress_callback
        )

        # 加载到内存
        self.raw_data = pd.read_csv(filepath, encoding='utf-8-sig')

        elapsed = time.time() - start_time
        logger.info(f"数据获取完成，耗时: {elapsed:.1f}秒")
        logger.info(f"数据文件: {filepath}")

        return filepath

    def run_filter(self, raw_data_path: str = None,
                   top_list: list = [100, 200],
                   **filter_kwargs) -> dict:
        """
        执行筛选评分流程

        Args:
            raw_data_path: 原始数据文件路径（为None则使用内存中的数据）
            top_list: 要导出的Top N列表
            **filter_kwargs: 筛选参数

        Returns:
            导出文件路径字典
        """
        logger.info("=" * 60)
        logger.info("开始执行筛选评分流程")
        logger.info("=" * 60)

        start_time = time.time()

        # 加载原始数据
        if raw_data_path:
            self.raw_data = pd.read_csv(raw_data_path, encoding='utf-8-sig')
        elif self.raw_data is None:
            # 尝试加载最新的原始数据文件
            raw_data_dir = Path(self.fetcher.raw_data_dir)
            raw_files = list(raw_data_dir.glob("all_stocks_*.csv"))
            if raw_files:
                latest_file = max(raw_files, key=lambda x: x.stat().st_mtime)
                self.raw_data = pd.read_csv(latest_file, encoding='utf-8-sig')
                logger.info(f"加载原始数据: {latest_file}")
            else:
                raise ValueError("未找到原始数据文件，请先运行 --fetch-only")

        # 执行筛选和导出
        exported_files = self.filter.process_and_export(
            self.raw_data,
            top_list=top_list,
            **filter_kwargs
        )

        # 加载Top 100到内存
        if 'top_100' in exported_files:
            top100_path = exported_files['top_100']
            self.filtered_data = pd.read_csv(top100_path, encoding='utf-8-sig')

        elapsed = time.time() - start_time
        logger.info(f"筛选评分完成，耗时: {elapsed:.1f}秒")

        return exported_files

    def run_report(self, top_n: int = 100,
                   filtered_data_path: str = None) -> list:
        """
        执行报告生成流程

        Args:
            top_n: 生成报告的股票数量
            filtered_data_path: 筛选后的数据文件路径

        Returns:
            生成的报告文件路径列表
        """
        logger.info("=" * 60)
        logger.info("开始执行报告生成流程")
        logger.info("=" * 60)

        start_time = time.time()

        # 加载筛选后的数据
        if filtered_data_path:
            self.filtered_data = pd.read_csv(filtered_data_path, encoding='utf-8-sig')
        elif self.filtered_data is None:
            # 尝试加载最新的Top 100文件
            output_dir = Path(self.filter.output_dir)
            top_files = list(output_dir.glob("top_100_*.csv"))
            if top_files:
                latest_file = max(top_files, key=lambda x: x.stat().st_mtime)
                self.filtered_data = pd.read_csv(latest_file, encoding='utf-8-sig')
                logger.info(f"加载筛选数据: {latest_file}")
            else:
                raise ValueError("未找到筛选数据文件，请先运行 --filter-only")

        # 生成报告
        report_files = self.reporter.generate_all_reports(
            self.filtered_data,
            limit=top_n
        )

        # 生成索引文件
        index_path = self.reporter.generate_summary_index(
            report_files,
            self.filtered_data
        )

        # 导出JSON格式
        json_path = self.reporter.export_json_format(
            self.filtered_data,
            limit=top_n
        )

        elapsed = time.time() - start_time
        logger.info(f"报告生成完成，耗时: {elapsed:.1f}秒")
        logger.info(f"共生成 {len(report_files)} 份报告")

        return report_files

    def run_full(self, years: int = 5, top_n: int = 100,
                 top_list: list = [100, 200], sample: int = None,
                 **filter_kwargs) -> dict:
        """
        执行完整流程

        Args:
            years: 财务数据年限
            top_n: 生成报告的股票数量
            top_list: 要导出的Top N列表
            **filter_kwargs: 筛选参数

        Returns:
            所有输出文件路径字典
        """
        logger.info("=" * 60)
        logger.info("开始执行完整分析流程 (V2.0)")
        logger.info("=" * 60)

        overall_start = time.time()

        all_files = {}

        # 1. 数据获取
        raw_file = self.run_fetch(years=years, sample=sample)
        all_files['raw_data'] = raw_file

        # 2. 筛选评分
        filtered_files = self.run_filter(top_list=top_list, **filter_kwargs)
        all_files.update(filtered_files)

        # 3. 报告生成
        report_files = self.run_report(top_n=top_n)
        all_files['reports'] = report_files

        overall_elapsed = time.time() - overall_start

        logger.info("=" * 60)
        logger.info("完整分析流程完成！")
        logger.info("=" * 60)
        logger.info(f"总耗时: {overall_elapsed:.1f}秒")
        logger.info("")
        logger.info("输出文件汇总:")
        logger.info(f"  1. 原始数据:   {raw_file}")
        logger.info(f"  2. Top 100:    {filtered_files.get('top_100', 'N/A')}")
        logger.info(f"  3. Top 200:    {filtered_files.get('top_200', 'N/A')}")
        logger.info(f"  4. 报告数量:   {len(report_files)} 份")
        logger.info(f"  5. 报告目录:   {self.reporter.output_dir}")

        return all_files

    def _progress_callback(self, current: int, total: int, ts_code: str):
        """进度回调函数"""
        if current % 50 == 0 or current == 1 or current == total:
            logger.info(f"获取进度: {current}/{total} ({current/total*100:.1f}%) - {ts_code}")

    def print_summary(self):
        """打印分析结果摘要"""
        if self.filtered_data is None or self.filtered_data.empty:
            logger.warning("无分析结果可显示")
            return

        print("\n" + "=" * 70)
        print("分析结果摘要")
        print("=" * 70)

        # 评级分布
        if 'rating' in self.filtered_data.columns:
            print("\n评级分布:")
            rating_counts = self.filtered_data['rating'].value_counts()
            for rating, count in rating_counts.items():
                print(f"  {rating}: {count} 只")

        # 行业分布（Top 10）
        if 'industry' in self.filtered_data.columns:
            print("\n行业分布 (Top 10):")
            industry_counts = self.filtered_data['industry'].value_counts().head(10)
            for industry, count in industry_counts.items():
                print(f"  {industry}: {count} 只")

        # Top 10股票
        print("\nTop 10 股票:")
        top10 = self.filtered_data.head(10)[
            ['rank', 'ts_code', 'name', 'industry', 'total_score', 'rating', 'pe', 'roe']
        ]
        print(top10.to_string(index=False))

        print("\n" + "=" * 70)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='A股基本面分析器 V2.0 - 分级输出版本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main_v2.py --full              # 完整流程
  python main_v2.py --fetch-only        # 仅获取数据
  python main_v2.py --filter-only       # 仅筛选评分
  python main_v2.py --report-only       # 仅生成报告
  python main_v2.py --fetch --filter    # 获取数据+筛选评分

配置:
  在config.py中设置TUSHARE_TOKEN
        """
    )

    # 流程控制参数
    parser.add_argument('--full', action='store_true',
                       help='执行完整分析流程（数据获取+筛选+报告）')
    parser.add_argument('--fetch-only', action='store_true',
                       help='仅执行数据获取流程')
    parser.add_argument('--filter-only', action='store_true',
                       help='仅执行筛选评分流程（需要已有原始数据）')
    parser.add_argument('--report-only', action='store_true',
                       help='仅执行报告生成流程（需要已有筛选数据）')
    parser.add_argument('--fetch', action='store_true',
                       help='包含数据获取流程')
    parser.add_argument('--filter', action='store_true',
                       help='包含筛选评分流程')
    parser.add_argument('--report', action='store_true',
                       help='包含报告生成流程')

    # 参数配置
    parser.add_argument('--years', type=int, default=5,
                       help='财务数据年限（默认5年）')
    parser.add_argument('--top-n', type=int, default=100,
                       help='生成报告的股票数量（默认100）')
    parser.add_argument('--top-list', type=int, nargs='+', default=[100, 200],
                       help='要导出的Top N列表（默认100 200）')
    parser.add_argument('--min-data-years', type=int, default=3,
                       help='最少数据年限要求（默认3年）')
    parser.add_argument('--exclude-st', action='store_true', default=True,
                       help='排除ST股票（默认True）')
    parser.add_argument('--no-exclude-st', action='store_false', dest='exclude_st',
                       help='不排除ST股票')
    parser.add_argument('--exclude-financials', action='store_true',
                       help='排除金融股（银行、保险、证券等）')
    parser.add_argument('--raw-data', type=str,
                       help='指定原始数据文件路径')
    parser.add_argument('--filtered-data', type=str,
                       help='指定筛选数据文件路径')
    parser.add_argument('--token', type=str,
                       help='Tushare Pro API Token（覆盖配置文件）')
    parser.add_argument('--show-summary', action='store_true',
                       help='显示分析结果摘要')
    parser.add_argument('--sample', type=int, default=None,
                       help='只获取前N只股票（用于测试）')

    args = parser.parse_args()

    # 如果没有指定任何操作，显示帮助
    if not any([args.full, args.fetch_only, args.filter_only,
                args.report_only, args.fetch, args.filter, args.report]):
        parser.print_help()
        return

    print("=" * 70)
    print("A股基本面分析器 V2.0 - 分级输出版本")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # 初始化分析器
        analyzer = StockAnalyzerV2(token=args.token)

        # 执行相应流程
        if args.full:
            # 完整流程
            analyzer.run_full(
                years=args.years,
                top_n=args.top_n,
                top_list=args.top_list,
                sample=args.sample,
                min_data_years=args.min_data_years,
                exclude_st=args.exclude_st,
                exclude_financials=args.exclude_financials
            )

        else:
            # 组合流程
            if args.fetch or args.fetch_only:
                analyzer.run_fetch(years=args.years, sample=args.sample)

            if args.filter or args.filter_only:
                analyzer.run_filter(
                    raw_data_path=args.raw_data,
                    top_list=args.top_list,
                    min_data_years=args.min_data_years,
                    exclude_st=args.exclude_st,
                    exclude_financials=args.exclude_financials
                )

            if args.report or args.report_only:
                analyzer.run_report(
                    top_n=args.top_n,
                    filtered_data_path=args.filtered_data
                )

        # 显示摘要
        if args.show_summary or args.full or args.filter_only:
            analyzer.print_summary()

        print("\n" + "=" * 70)
        print("分析完成！")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n用户中断，程序退出")
        sys.exit(0)
    except Exception as e:
        logger.error(f"程序运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
