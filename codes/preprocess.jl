# 导入所需的包
using CSV
using DataFrames
using Statistics

# 获取 Julia 脚本的目录
const SCRIPT_DIR = @__DIR__

# 设置文件路径
input_file_path = abspath(joinpath(SCRIPT_DIR, "../question/男胎检测数据.csv"))
output_file_path = abspath(joinpath(SCRIPT_DIR, "../output/processed_data.csv"))

function process_fetal_data(input_path::String, output_path::String)
    df = CSV.read(input_path, DataFrame, missingstring="", normalizenames=true)
    # 列名标准化
    DataFrames.rename!(df, Symbol("孕妇代码") => :code,
        Symbol("IVF妊娠") => :ivf_pregnancy,
        Symbol("年龄") => :age,
        Symbol("检测抽血次数") => :blood_draw_count,
        Symbol("检测孕周") => :gestational_week,
        Symbol("孕妇BMI") => :bmi,
        Symbol("怀孕次数") => :pregnancy_count,
        Symbol("生产次数") => :birth_count,
        Symbol("Y染色体浓度") => :y_chromosome_concentration)
    # 转换妊娠类型为数值 
    replace!(df.ivf_pregnancy, "自然受孕" => "1", "试管婴儿" => "2", "人工授精" => "3")
    # 转换“>=3”为3
    replace!(df.pregnancy_count, "≥3" => "3")
    df.pregnancy_count = parse.(Int64, df.pregnancy_count)
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
    # 按孕妇代码和检测抽血次数分组，并计算Y染色体浓度的平均值
    grouped_df = combine(groupby(df, [:code, :blood_draw_count]),
        :age => first => :age,
        :ivf_pregnancy => first => :ivf_pregnancy,
        :gestational_week => first => :gestational_week,
        :bmi => first => :bmi,
        :pregnancy_count => first => :pregnancy_count,
        :birth_count => first => :birth_count,
        :y_chromosome_concentration => mean => :y_chromosome_concentration)
    sort!(grouped_df, [:code, :blood_draw_count])
    # 重新排列列的顺序、重命名
    select!(grouped_df, :code, :ivf_pregnancy, :age, :blood_draw_count,
        :gestational_week, :bmi, :pregnancy_count,
        :birth_count, :y_chromosome_concentration)
    DataFrames.rename!(grouped_df, :code => "孕妇代码",
        :ivf_pregnancy => "IVF妊娠",
        :age => "年龄",
        :blood_draw_count => "检测抽血次数",
        :gestational_week => "检测孕周",
        :bmi => "孕妇BMI",
        :pregnancy_count => "怀孕次数",
        :birth_count => "生产次数",
        :y_chromosome_concentration => "Y染色体浓度")
    CSV.write(output_path, grouped_df)
end

# 调用函数执行处理
process_fetal_data(input_file_path, output_file_path)
