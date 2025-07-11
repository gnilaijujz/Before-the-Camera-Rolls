# Before the Camera Rolls
Web Mining Project - SOC Summer Workshop 2025
main函数：
1.数据处理：

（1）简单清洗数据，包括处理缺失值、时间格式的转换（使用data_processing文件的函数load_and_clean_data）；

（2）归一化的热度按照区间分段，目前是7天一分段，后期可以调整，新增period_label列，相同分段的period_label应该是一致的（使用data_processing文件的函数split_time_periods）

（3）给不同分段的函数赋了相应的权重，依照的是指数分布，然后用该权重乘了各数据项的归一化热度，新增heat列，heat是更具有时间效益的数据，后面所有模型的衡量指标都可以直接拿heat这一列进行分析（使用weighting文件的add_heat_column函数）

2.数据分析

在此因为我分析的是上传时间，所以以此为例。

前面的data_processing文件、weighting文件都是前面用于数据处理通用的文件，还有个utils可能是个日志管理相关的文件？

而publish_time_analyzer就是针对上传时间的分析代码所在的文件，所以每个人可以把自己写的那部分分析代码放在一个像这样的单独文件里，后面可以在main函数里调用用于分析数据

3.后面的LLM大语言模型的接口还没有做，所以先注释掉了，后面会尝试。
目前为了便于调试就是在终端可以显示我的数据分析结果。


其实就是把自己那部分数据分析代码放在该项目文件夹下，封装成函数即可，注意衡量指标是用处理过的heat值，其中有任何问题，或者无法理解的方面咱们再讨论。
