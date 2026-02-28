# A股基本面分析器 - 快速开始

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

## 2. 运行测试

```bash
python test.py
```

## 3. 快速运行

### 测试模式（100只股票，约5分钟）

```bash
cd src
python main.py --sample 100
```

### 完整分析（全部A股，约2-3小时）

```bash
cd src
python main.py
```

## 4. 断点续传

如果程序中断，重新运行即可从断点继续：

```bash
python main.py
```

如需重新开始：

```bash
python main.py --restart
```

查看当前进度：

```bash
python main.py --progress
```

## 5. 查看结果

分析完成后，结果保存在 `output` 目录：

- `a_stock_analysis.xlsx` - 全部股票分析结果（含每年PE/PB/股息率）
- `comprehensive_report.xlsx` - 综合报告（多工作表）
- `summary_report.txt` - 汇总统计信息

## 常见问题

**Q: 提示"积分不足"怎么办？**  
A: 访问 https://tushare.pro 完善资料获取积分，或购买积分套餐。

**Q: 程序中断后如何继续？**  
A: 直接重新运行 `python main.py` 即可从断点自动恢复。

**Q: 如何重新开始？**  
A: 运行 `python main.py --restart` 或删除 `data/checkpoint.pkl` 文件。

**Q: 输出文件会被覆盖吗？**  
A: 是的，程序使用单版本模式，每次运行会覆盖之前的输出文件。
