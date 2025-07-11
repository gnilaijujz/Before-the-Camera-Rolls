# main.py
print("=== 程序开始运行 ===")
from data_processing import load_and_clean_data
from data_processing import split_time_periods
from publish_time_analyzer import analyze_publish_time
# from analysis.content_tag_analyzer import analyze_content_tags  # 注释未实现部分
#from result_aggregator.aggregator import aggregate_results
#from llm_integration.llm_client import llm_integrate
# from config.settings import LLM_CONFIG, DATA_PATH  # 暂时不依赖配置文件，直接定义路径
from utils.logger import get_logger
from weighting import add_heat_column  # 加权热度计算

import json

# 手动定义数据文件路径（核心修改：替换为你的数据文件实际路径）
DATA_PATH = "sports_videos.csv"  # 例如：如果数据在data文件夹下，改为 "data/sports_videos.csv"
LLM_CONFIG = {"api_key": ""}  # 暂时留空，不影响当前调试

print("1. 开始执行 main 函数")

def main():
   
    print("2. 进入 main 函数")
    logger = get_logger()
    logger.info("项目启动：开始分析体育类视频数据")
   
    # 1. 数据处理
    # 1. 数据处理（拆分步骤，详细报错）
    try:
        print("3. 准备调用 load_and_clean_data")
        if not DATA_PATH:
            raise ValueError("DATA_PATH 未定义，请检查数据文件路径配置")
    
    # 调用数据加载函数
        df = load_and_clean_data(DATA_PATH)
        print("4. load_and_clean_data 执行完成")
    
    # 检查数据加载结果
        if df.empty:
           raise ValueError("load_and_clean_data 返回空数据，请检查数据源是否有效")
        print(f"5. 数据加载成功，原始数据量：{len(df)} 行")

    except Exception as e:
        logger.error(f"数据加载阶段失败：{str(e)}", exc_info=True)  # 打印完整错误堆栈
        return

    try:
    # 步骤2：划分时间段
       print("6. 准备调用 split_time_periods")
    # 明确传递时间列参数
       df, period_map = split_time_periods(
           df,
           time_col="published_at",  # 确保与数据中的时间列名一致
           interval=7
       )
       print("7. split_time_periods 执行完成")
    
    # 检查时间段划分结果
       if "period_label" not in df.columns:  # 假设函数会添加period_label列
           raise ValueError("split_time_periods 未正确添加时间段标签列")
       logger.info(f"数据处理完成，样本量：{len(df)}，时间段数量：{len(period_map)}")

    except Exception as e:
       logger.error(f"时间段划分阶段失败：{str(e)}", exc_info=True)  # 打印完整错误堆栈
       return
    
    # 3. 用新的weighting计算heat列（基于已有的period_label）
  
    try:
    
        df = add_heat_column(
            df=df,
            heat_col="popularity_normalized",  # 例如修正为这个列名
            decay_rate=0.8,
            min_samples=5
        )
        print("heat 列计算完成")
    except ValueError as e:
        print(f"计算 heat 列失败：{e}")
        return

    # 2. 多维度分析（仅保留发布时间分析）
    try:
      
        time_result = analyze_publish_time(df)
        analysis_results = [time_result]  # 仅保留时间分析结果
        logger.info("发布时间分析完成")
    except Exception as e:
        logger.error(f"分析失败：{str(e)}")
        return

    # 3. 打印时间分析结果（简化输出）
    print("\n发布时间分析结果：")
    print(json.dumps(time_result, indent=2, ensure_ascii=False, default=str))
  
    # 注释未实现的部分
    # aggregated_data = aggregate_results(
    #     analysis_results, 
    #     business_context="体育类视频最佳发布时间与内容策略分析"
    # )

    # integrated_content = llm_integrate(
    #     aggregated_data, 
    #     api_key=LLM_CONFIG["api_key"]
    # )

    # with open("analysis_report.txt", "w", encoding="utf-8") as f:
    #     f.write(integrated_content)
    # logger.info("分析报告已保存至 analysis_report.txt")
    # print("\n整合分析结果：\n", integrated_content)

if __name__ == "__main__":
    main()