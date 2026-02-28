"""
A股基本面分析器 - 主程序入口

功能：
1. 使用Tushare Pro API获取A股全量数据
2. 分析每只股票最近三年的基本面指标（PE、PB、股息率等）
3. 计算统计值、趋势和价值评分
4. 导出分析结果为Excel/CSV格式

特性：
- 断点保护：每100只股票自动保存进度，支持断点续传
- 单版本输出：只保留最新分析结果
- 自动恢复：程序中断后可从上次位置继续

使用方法：
1. 在config.py中填入你的Tushare Pro API Token
2. 运行: python main.py
3. 查看output目录中的分析结果

断点续传：
- 程序会自动保存进度到data/checkpoint.pkl
- 中断后重新运行会自动从上次位置继续
- 如需重新开始，删除data/checkpoint.pkl即可
"""

import sys
import os
import time
import argparse
import logging
import pickle
import signal
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

# 导入自定义模块
from config import (
    TUSHARE_TOKEN, ANALYSIS_YEARS, DATA_DIR, OUTPUT_DIR, LOG_DIR,
    API_CALL_DELAY, API_MAX_RETRY, CHECKPOINT_INTERVAL
)
from data_fetcher import TushareDataFetcher
from data_validator import DataValidator, validate_analysis_result
from analyzer import FundamentalAnalyzer, filter_stocks_by_criteria
from exporter import DataExporter

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, f'analyzer_{datetime.now().strftime("%Y%m%d")}.log'), encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 全局变量用于信号处理
_g_analyzer = None


def signal_handler(signum, frame):
    """信号处理函数 - 捕获Ctrl+C等中断信号"""
    logger.warning(f"收到信号 {signum}，正在保存进度...")
    if _g_analyzer is not None:
        _g_analyzer._save_checkpoint()
    logger.info("进度已保存，程序退出")
    sys.exit(0)


# 注册信号处理
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class CheckpointManager:
    """断点管理器 - 负责保存和恢复分析进度"""
    
    def __init__(self, checkpoint_dir: str = DATA_DIR):
        self.checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pkl")
        self.state = {
            "processed_stocks": set(),  # 已处理的股票代码集合
            "stock_list": None,         # 股票列表
            "stock_info_dict": {},      # 股票信息字典
            "historical_data": {},      # 历史数据（部分）
            "analysis_results": [],     # 分析结果列表
            "start_time": None,         # 开始时间
            "last_update": None,        # 最后更新时间
            "total_stocks": 0,          # 总股票数
            "config": {                 # 配置信息
                "years": ANALYSIS_YEARS,
                "token_prefix": TUSHARE_TOKEN[:10] if TUSHARE_TOKEN else ""
            }
        }
    
    def exists(self) -> bool:
        """检查是否存在断点文件"""
        return os.path.exists(self.checkpoint_path)
    
    def save(self, **kwargs):
        """保存断点状态"""
        self.state["last_update"] = datetime.now().isoformat()
        for key, value in kwargs.items():
            if key in self.state:
                self.state[key] = value
        
        try:
            with open(self.checkpoint_path, 'wb') as f:
                pickle.dump(self.state, f)
            logger.info(f"断点已保存: {len(self.state['processed_stocks'])}/{self.state['total_stocks']} 只股票")
        except Exception as e:
            logger.error(f"保存断点失败: {e}")
    
    def load(self) -> Dict:
        """加载断点状态"""
        if not self.exists():
            return None
        
        try:
            with open(self.checkpoint_path, 'rb') as f:
                self.state = pickle.load(f)
            
            # 验证配置是否匹配
            saved_config = self.state.get("config", {})
            if saved_config.get("years") != ANALYSIS_YEARS:
                logger.warning("配置不匹配（分析年限），将重新开始")
                return None
            
            logger.info(f"断点已加载: {len(self.state['processed_stocks'])}/{self.state['total_stocks']} 只股票已处理")
            return self.state
        except Exception as e:
            logger.error(f"加载断点失败: {e}")
            return None
    
    def clear(self):
        """清除断点"""
        if self.exists():
            os.remove(self.checkpoint_path)
            logger.info("断点已清除")
    
    def get_progress(self) -> tuple:
        """获取进度信息"""
        processed = len(self.state.get("processed_stocks", set()))
        total = self.state.get("total_stocks", 0)
        return processed, total


class AStockAnalyzer:
    """A股基本面分析器主类（支持断点保护）"""
    
    def __init__(self, token: str = None, years: int = ANALYSIS_YEARS):
        """
        初始化分析器
        
        Args:
            token: Tushare Pro API Token
            years: 分析年限
        """
        self.token = token or TUSHARE_TOKEN
        self.years = years
        
        # 检查token
        if not self.token or self.token == "your_token_here":
            raise ValueError(
                "请提供有效的Tushare Pro API Token！\n"
                "1. 访问 https://tushare.pro/register 注册账号\n"
                "2. 获取Token后在config.py中设置，或通过环境变量TUSHARE_TOKEN传入"
            )
        
        # 初始化组件
        logger.info("正在初始化组件...")
        self.fetcher = TushareDataFetcher(self.token)
        self.validator = DataValidator()
        self.analyzer = FundamentalAnalyzer(years)
        self.exporter = DataExporter()
        
        # 初始化断点管理器
        self.checkpoint = CheckpointManager()
        
        # 数据存储
        self.stock_list = None
        self.stock_info_dict = {}
        self.historical_data = {}
        self.analysis_results = []
        
        # 进度跟踪
        self.processed_stocks = set()
        self.start_time = None
        
        logger.info(f"AStockAnalyzer 初始化完成，分析年限: {years}年")
    
    def _save_checkpoint(self):
        """保存当前进度"""
        self.checkpoint.save(
            processed_stocks=self.processed_stocks,
            stock_list=self.stock_list,
            stock_info_dict=self.stock_info_dict,
            historical_data=self.historical_data,
            analysis_results=self.analysis_results,
            start_time=self.start_time,
            total_stocks=len(self.stock_list) if self.stock_list is not None else 0
        )
    
    def _load_checkpoint(self) -> bool:
        """加载断点，返回是否成功恢复"""
        state = self.checkpoint.load()
        if state is None:
            return False
        
        self.processed_stocks = state.get("processed_stocks", set())
        self.stock_list = state.get("stock_list")
        self.stock_info_dict = state.get("stock_info_dict", {})
        self.historical_data = state.get("historical_data", {})
        self.analysis_results = state.get("analysis_results", [])
        self.start_time = state.get("start_time")
        
        return True
    
    def fetch_stock_list(self, list_status: str = 'L', force_refresh: bool = False) -> pd.DataFrame:
        """
        获取股票列表
        
        Args:
            list_status: 上市状态，L=上市，D=退市，P=暂停上市
            force_refresh: 强制刷新，忽略缓存
            
        Returns:
            股票列表DataFrame
        """
        # 如果已加载断点，直接使用
        if self.stock_list is not None and not force_refresh:
            logger.info(f"使用已加载的股票列表: {len(self.stock_list)} 只")
            return self.stock_list
        
        logger.info("正在获取股票列表...")
        self.stock_list = self.fetcher.get_all_stocks(list_status)
        
        # 验证数据
        validation = self.validator.validate_stock_list(self.stock_list)
        if not validation["is_valid"]:
            logger.error(f"股票列表验证失败: {validation['errors']}")
            raise ValueError("股票列表数据无效")
        
        # 构建股票信息字典
        for _, row in self.stock_list.iterrows():
            self.stock_info_dict[row["ts_code"]] = row.to_dict()
        
        logger.info(f"成功获取 {len(self.stock_list)} 只股票")
        return self.stock_list
    
    def fetch_historical_data(self,
                             start_date: str = None,
                             end_date: str = None,
                             sample_size: int = None,
                             resume: bool = True,
                             use_yearly: bool = True) -> Dict[str, pd.DataFrame]:
        """
        获取历史数据（支持断点续传）

        Args:
            start_date: 开始日期 (YYYYMMDD)，默认计算years年前的日期（仅当日线模式时有效）
            end_date: 结束日期 (YYYYMMDD)，默认今天（仅当日线模式时有效）
            sample_size: 抽样数量（用于测试），None表示获取全部
            resume: 是否尝试从断点恢复
            use_yearly: 是否使用年度数据模式，True=获取年度财务指标，False=获取日线数据

        Returns:
            股票代码到数据的映射字典
        """
        if self.stock_list is None or self.stock_list.empty:
            raise ValueError("请先调用fetch_stock_list()获取股票列表")

        # 根据数据模式计算参数
        if use_yearly:
            # 年度数据模式：计算年份范围
            current_year = datetime.now().year
            start_year = current_year - self.years
            end_year = current_year - 1  # 不包含当前年份（年报可能未出）
            logger.info(f"数据获取模式: 年度财务数据 ({start_year}-{end_year})")
        else:
            # 日线数据模式：计算日期范围
            if end_date is None:
                end_date = datetime.now().strftime('%Y%m%d')

            if start_date is None:
                start = datetime.now() - timedelta(days=365 * self.years)
                start_date = start.strftime('%Y%m%d')

            logger.info(f"数据获取模式: 日线数据 ({start_date} 至 {end_date})")

        # 确定要处理的股票列表
        stocks_to_fetch = self.stock_list.copy()
        if sample_size and sample_size < len(stocks_to_fetch):
            stocks_to_fetch = stocks_to_fetch.sample(n=sample_size, random_state=42)
            logger.info(f"抽样模式: 只获取 {sample_size} 只股票的数据")

        total = len(stocks_to_fetch)
        logger.info(f"总共需要处理: {total} 只股票")

        # 获取交易日历（仅日线模式需要）
        trade_cal = None
        if not use_yearly:
            logger.info("获取交易日历...")
            trade_cal = self.fetcher.get_trade_calendar(start_date, end_date)
            if trade_cal.empty:
                logger.warning("无法获取交易日历")
            else:
                logger.info(f"共有 {len(trade_cal)} 个交易日")

        # 过滤已处理的股票
        if resume and self.processed_stocks:
            stocks_to_fetch = stocks_to_fetch[~stocks_to_fetch['ts_code'].isin(self.processed_stocks)]
            logger.info(f"已处理 {len(self.processed_stocks)} 只，剩余 {len(stocks_to_fetch)} 只需要处理")

        # 获取数据
        success_count = len(self.processed_stocks)
        failed_stocks = []
        incomplete_stocks = []
        batch_count = 0

        data_mode = "年度" if use_yearly else "日线"
        logger.info(f"开始获取{data_mode}数据...")

        for i, (_, stock) in enumerate(stocks_to_fetch.iterrows()):
            ts_code = stock["ts_code"]
            current_idx = success_count + i + 1

            # 每100只保存一次断点
            if i > 0 and i % CHECKPOINT_INTERVAL == 0:
                self._save_checkpoint()
                batch_count += 1
                logger.info(f"=== 已处理 {current_idx}/{total} 只，保存断点 ===")

            if i % 10 == 0:
                logger.info(f"进度: {current_idx}/{total} ({current_idx/total*100:.1f}%)，成功: {success_count}")

            try:
                if use_yearly:
                    # 使用年度财务数据模式
                    df = self.fetcher.get_yearly_fina_data(
                        ts_code=ts_code,
                        start_year=start_year,
                        end_year=end_year
                    )

                    if df.empty:
                        failed_stocks.append((ts_code, "无年度数据"))
                        continue

                    # 验证年度数据
                    validation = self.validator.validate_yearly_data(df, ts_code, self.years)

                    if not validation["is_valid"]:
                        logger.warning(f"{ts_code} 年度数据验证失败: {validation['errors']}")
                        if validation["actual_years"] > 0:
                            # 数据不完整但仍有价值，保留但记录
                            incomplete_stocks.append((ts_code, f"年份数不足: {validation['actual_years']}/{self.years}"))
                            self.historical_data[ts_code] = df
                            self.processed_stocks.add(ts_code)
                        else:
                            failed_stocks.append((ts_code, "验证失败"))
                        continue

                    # 检查数据完整性
                    if validation["actual_years"] < self.years:
                        incomplete_stocks.append((ts_code, f"年份数不足: {validation['actual_years']}/{self.years}"))
                        logger.info(f"{ts_code}: 获取到 {validation['actual_years']} 年数据")
                    else:
                        logger.info(f"{ts_code}: 获取到完整 {validation['actual_years']} 年数据")

                    # 存储数据
                    self.historical_data[ts_code] = df
                    self.processed_stocks.add(ts_code)
                    success_count += 1

                else:
                    # 使用原有日线数据模式
                    df = self.fetcher.get_daily_basic_by_stock(
                        ts_code=ts_code,
                        start_date=start_date,
                        end_date=end_date,
                        use_yearly=False
                    )

                    if df.empty:
                        failed_stocks.append((ts_code, "无数据"))
                        continue

                    # 验证数据
                    year = datetime.now().year
                    validation = self.validator.validate_daily_basic(df, ts_code, year)

                    if not validation["is_valid"]:
                        logger.warning(f"{ts_code} 数据验证失败: {validation['errors']}")
                        failed_stocks.append((ts_code, "验证失败"))
                        continue

                    # 清洗数据
                    df_clean = self.validator.clean_data(df, ts_code)

                    # 存储数据
                    self.historical_data[ts_code] = df_clean
                    self.processed_stocks.add(ts_code)
                    success_count += 1

            except Exception as e:
                logger.error(f"{ts_code} 数据获取失败: {e}")
                failed_stocks.append((ts_code, str(e)))
                continue

        # 最终保存
        self._save_checkpoint()

        logger.info(f"数据获取完成: 成功 {success_count}/{total}")

        # 输出数据完整性报告
        if use_yearly:
            completeness_report = self.validator.check_data_completeness(
                self.historical_data, self.years
            )
            logger.info(f"数据完整性报告:")
            logger.info(f"  - 完整数据({self.years}年): {completeness_report['complete_stocks']} 只")
            logger.info(f"  - 不完整数据: {completeness_report['incomplete_stocks']} 只")
            logger.info(f"  - 无数据: {completeness_report['missing_stocks']} 只")
            if completeness_report['year_distribution']:
                logger.info(f"  - 年份数分布: {completeness_report['year_distribution']}")

        if failed_stocks:
            logger.warning(f"失败股票数量: {len(failed_stocks)}")
            failed_df = pd.DataFrame(failed_stocks, columns=["ts_code", "reason"])
            failed_path = os.path.join(OUTPUT_DIR, "failed_stocks.csv")
            failed_df.to_csv(failed_path, index=False, encoding='utf-8-sig')

        if incomplete_stocks:
            logger.info(f"不完整股票数量: {len(incomplete_stocks)}")
            incomplete_df = pd.DataFrame(incomplete_stocks, columns=["ts_code", "reason"])
            incomplete_path = os.path.join(OUTPUT_DIR, "incomplete_stocks.csv")
            incomplete_df.to_csv(incomplete_path, index=False, encoding='utf-8-sig')

        return self.historical_data
    
    def analyze(self, resume: bool = True) -> pd.DataFrame:
        """
        执行分析（支持断点续传）
        
        Args:
            resume: 是否尝试从断点恢复
            
        Returns:
            分析结果DataFrame
        """
        if not self.historical_data:
            raise ValueError("请先调用fetch_historical_data()获取历史数据")
        
        # 确定需要分析的股票
        stocks_to_analyze = {}
        already_analyzed = set()
        
        if resume and self.analysis_results:
            already_analyzed = {r.get("ts_code") for r in self.analysis_results if r.get("ts_code")}
            logger.info(f"已有 {len(already_analyzed)} 只股票的分析结果")
        
        for ts_code, df in self.historical_data.items():
            if ts_code not in already_analyzed:
                stocks_to_analyze[ts_code] = df
        
        total = len(stocks_to_analyze)
        logger.info(f"开始分析 {total} 只股票...")
        
        # 批量分析
        for i, (ts_code, df) in enumerate(stocks_to_analyze.items()):
            if i % 100 == 0:
                logger.info(f"分析进度: {i}/{total} ({i/total*100:.1f}%)")
            
            stock_info = self.stock_info_dict.get(ts_code, {})
            result = self.analyzer.analyze_stock(df, stock_info)
            self.analysis_results.append(result)
            
            # 每100只保存一次
            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                self._save_checkpoint()
                logger.info(f"=== 已分析 {i+1}/{total} 只，保存断点 ===")
        
        # 最终保存
        self._save_checkpoint()
        
        # 转换为DataFrame
        self.analysis_results = self.analyzer._results_to_dataframe(self.analysis_results)
        
        logger.info(f"分析完成，共 {len(self.analysis_results)} 只股票")
        return self.analysis_results
    
    def export_results(self) -> List[str]:
        """
        导出分析结果（单版本模式）
        
        Returns:
            导出的文件路径列表
        """
        if self.analysis_results is None or (isinstance(self.analysis_results, list) and not self.analysis_results):
            raise ValueError("请先调用analyze()执行分析")
        
        if isinstance(self.analysis_results, list):
            self.analysis_results = self.analyzer._results_to_dataframe(self.analysis_results)
        
        exported_files = []
        
        logger.info("开始导出结果...")
        
        # 导出为固定文件名的Excel（单版本）
        excel_path = self.exporter.export_to_excel(
            self.analysis_results, 
            filename="a_stock_analysis",
            sheet_name="分析结果"
        )
        exported_files.append(excel_path)
        
        # 导出为固定文件名的CSV
        csv_path = self.exporter.export_to_csv(
            self.analysis_results, 
            filename="a_stock_analysis"
        )
        exported_files.append(csv_path)
        
        # 导出综合报告
        comp_path = self.exporter.export_comprehensive_report(
            self.analysis_results,
            filename="comprehensive_report"
        )
        exported_files.append(comp_path)
        
        # 生成并保存汇总报告
        summary = self.exporter.generate_summary_report(self.analysis_results)
        summary_path = self.exporter.save_summary_to_text(
            summary, 
            filename="summary_report"
        )
        exported_files.append(summary_path)
        
        logger.info(f"导出完成，共 {len(exported_files)} 个文件")
        return exported_files
    
    def run_full_analysis(self,
                         sample_size: int = None,
                         list_status: str = 'L',
                         resume: bool = True,
                         clear_checkpoint: bool = False,
                         use_yearly: bool = True) -> pd.DataFrame:
        """
        运行完整分析流程（支持断点续传）

        Args:
            sample_size: 抽样数量（用于测试），None表示获取全部
            list_status: 上市状态
            resume: 是否尝试从断点恢复
            clear_checkpoint: 是否清除断点（重新开始）
            use_yearly: 是否使用年度数据模式，True=获取年度财务指标，False=获取日线数据

        Returns:
            分析结果DataFrame
        """
        self.start_time = time.time()

        # 如果需要清除断点
        if clear_checkpoint:
            self.checkpoint.clear()
            logger.info("已清除断点，将重新开始")
            resume = False

        # 尝试加载断点
        if resume and self._load_checkpoint():
            logger.info("已从断点恢复")

        try:
            # 1. 获取股票列表
            self.fetch_stock_list(list_status)

            # 2. 获取历史数据
            self.fetch_historical_data(sample_size=sample_size, resume=resume, use_yearly=use_yearly)

            # 3. 执行分析
            self.analyze(resume=resume)

            # 4. 导出结果
            self.export_results()

            elapsed = time.time() - self.start_time
            logger.info(f"完整分析流程完成，耗时: {elapsed:.1f}秒")

            # 完成后清除断点
            self.checkpoint.clear()

            return self.analysis_results

        except KeyboardInterrupt:
            logger.warning("用户中断，保存进度...")
            self._save_checkpoint()
            logger.info("进度已保存，可以稍后继续")
            raise
        except Exception as e:
            logger.error(f"分析流程失败: {e}")
            self._save_checkpoint()
            logger.info("已保存断点，可以稍后继续")
            raise
    
    def get_top_stocks(self, 
                      n: int = 50, 
                      sort_by: str = "价值评分",
                      min_score: int = None) -> pd.DataFrame:
        """获取排名靠前的股票"""
        if self.analysis_results is None:
            raise ValueError("请先执行分析")
        
        if isinstance(self.analysis_results, list):
            df = self.analyzer._results_to_dataframe(self.analysis_results)
        else:
            df = self.analysis_results.copy()
        
        if min_score is not None and "价值评分" in df.columns:
            df = df[df["价值评分"] >= min_score]
        
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=False)
        
        return df.head(n)
    
    def filter_stocks(self, criteria: Dict) -> pd.DataFrame:
        """根据条件筛选股票"""
        if self.analysis_results is None:
            raise ValueError("请先执行分析")
        
        if isinstance(self.analysis_results, list):
            df = self.analyzer._results_to_dataframe(self.analysis_results)
        else:
            df = self.analysis_results.copy()
        
        return filter_stocks_by_criteria(df, criteria)
    
    def get_progress(self) -> Dict:
        """获取当前进度信息"""
        processed, total = self.checkpoint.get_progress()
        return {
            "processed": processed,
            "total": total,
            "percentage": (processed / total * 100) if total > 0 else 0,
            "remaining": total - processed
        }


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='A股基本面分析器（支持断点续传）')
    parser.add_argument('--token', type=str, default=None, help='Tushare Pro API Token')
    parser.add_argument('--years', type=int, default=3, help='分析年限（默认3年）')
    parser.add_argument('--sample', type=int, default=None, help='抽样数量（用于测试）')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR, help='输出目录')
    parser.add_argument('--restart', action='store_true', help='重新开始（忽略断点）')
    parser.add_argument('--progress', action='store_true', help='只显示进度信息')
    parser.add_argument('--daily', action='store_true', help='使用日线数据模式（默认使用年度数据模式）')
    args = parser.parse_args()

    print("=" * 70)
    print("A股基本面分析器（支持断点续传）")
    print("=" * 70)
    print(f"分析年限: {args.years}年")
    print(f"数据模式: {'年度财务数据' if not args.daily else '日线数据'}")
    if args.sample:
        print(f"抽样模式: {args.sample}只股票")
    if args.restart:
        print("模式: 重新开始（忽略断点）")
    print("=" * 70)

    # 设置全局变量用于信号处理
    global _g_analyzer

    try:
        # 初始化分析器
        analyzer = AStockAnalyzer(token=args.token, years=args.years)
        _g_analyzer = analyzer

        # 如果只显示进度
        if args.progress:
            progress = analyzer.get_progress()
            print(f"\n当前进度: {progress['processed']}/{progress['total']} ({progress['percentage']:.1f}%)")
            print(f"剩余: {progress['remaining']} 只股票")
            return

        # 运行完整分析
        results = analyzer.run_full_analysis(
            sample_size=args.sample,
            clear_checkpoint=args.restart,
            use_yearly=not args.daily
        )

        # 显示汇总信息
        print("\n" + "=" * 70)
        print("分析完成！")
        print("=" * 70)
        print(f"分析股票数量: {len(results)}")

        if "评级" in results.columns:
            print("\n评级分布:")
            print(results["评级"].value_counts())

        if "所属行业" in results.columns:
            print("\n行业分布(Top10):")
            print(results["所属行业"].value_counts().head(10))

        # 显示高分股票
        if "价值评分" in results.columns:
            print("\n高分股票(Top10):")
            top10 = results.nlargest(10, "价值评分")[["股票代码", "股票名称", "所属行业", "PE_3年均值", "PB_3年均值", "股息率_3年均值", "价值评分", "评级"]]
            print(top10.to_string(index=False))

        print(f"\n结果已保存到: {args.output}")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n用户中断，进度已保存")
        print("重新运行程序可从断点继续")
        sys.exit(0)
    except Exception as e:
        logger.error(f"程序运行失败: {e}")
        print(f"\n错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
