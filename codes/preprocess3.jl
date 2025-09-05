using CSV
using DataFrames
using Statistics

# 设置文件路径
input_data = abspath(joinpath(@__DIR__, "../question/男胎检测数据.csv"))
output_file_path = abspath(joinpath(@__DIR__, "../output/processed_data3.csv"))

function process_fetal_data(input_data::String, output_path::String)
    df = CSV.read(input_data, DataFrame, missingstring="", normalizenames=true)
    # 列名标准化
    DataFrames.rename!(df, Symbol("孕妇代码") => :code,
        Symbol("年龄") => :age,
        Symbol("身高") => :height,
        Symbol("体重") => :weight,
        Symbol("检测抽血次数") => :blood_draw_count,
        Symbol("检测孕周") => :gestational_week,
        Symbol("孕妇BMI") => :bmi,
        Symbol("原始读段数") => :oringin_reads,
        Symbol("唯一比对的读段数") => :unique_aligned_reads,
        Symbol("被过滤掉读段数的比例") => :filtered_read_ratio,
        Symbol("重复读段的比例") => :duplicate_read_ratio,
        Symbol("在参考基因组上比对的比例") => :alignment_ratio_to_reference_genome,
        Symbol("Y染色体的Z值") => :z_value_of_y_chromosome,
        Symbol("Y染色体浓度") => :y_chromosome_concentration)
    # 使周数+天数归一到周数上
    function parse_gestational_week(s)
        m = match(r"([-+]?\d+(?:\.\d+)?)w?([+-]?\d+)?", s)
        weeks_str = m.captures[1]
        weeks = parse(Float64, weeks_str)
        days = 0.0
        if length(m.captures) > 1 && m.captures[2] !== nothing
            days_str = m.captures[2]
            days = parse(Float64, days_str)
        end
        return weeks + days / 7
    end
    df.gestational_week = parse_gestational_week.(df.gestational_week)
    output = filter(row -> abs(row.oringin_reads * (1 - row.filtered_read_ratio) * row.alignment_ratio_to_reference_genome * (1 - row.duplicate_read_ratio) - row.unique_aligned_reads) < 2, df)
    # 重新排列列的顺序、重命名
    select!(output, :code,
        :age,
        :height,
        :weight,
        :blood_draw_count,
        :gestational_week,
        :bmi,
        :oringin_reads,
        :unique_aligned_reads,
        :filtered_read_ratio,
        :duplicate_read_ratio,
        :alignment_ratio_to_reference_genome,
        :z_value_of_y_chromosome,
        :y_chromosome_concentration)
    DataFrames.rename!(output, :code => "孕妇代码",
        :age => "年龄",
        :height => "身高",
        :weight => "体重",
        :blood_draw_count => "检测抽血次数",
        :gestational_week => "检测孕周",
        :bmi => "孕妇BMI",
        :oringin_reads => "原始读段数",
        :unique_aligned_reads => "唯一比对的读段数",
        :filtered_read_ratio => "被过滤掉读段数的比例",
        :duplicate_read_ratio => "重复读段的比例",
        :alignment_ratio_to_reference_genome => "在参考基因组上比对的比例",
        :z_value_of_y_chromosome => "Y染色体的Z值",
        :y_chromosome_concentration => "Y染色体浓度")
    CSV.write(output_path, output)
end

# 调用函数执行处理
process_fetal_data(input_data, output_file_path)
