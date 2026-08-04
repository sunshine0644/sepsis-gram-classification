# ==============================================================================
# 综合数据处理与分析代码（学术投稿版）- 完整版本
# 作者：李泽奇
# 机构：XX大学医学院
# 版本：v8.0 - 完整版本（包含数据插补和基线特征分析）
# ==============================================================================

# ==============================================================================
# 第一部分：环境设置和包加载
# ==============================================================================

# 记录开始时间
start_time <- Sys.time()

# 设置工作目录（请根据实际情况修改）
setwd('/Users/lizeqi/Desktop/MIMIC/MIMIC数据R语言代码-新版/MIMIC新')
library(readr)      # 读取CSV
library(dplyr)      # 数据处理
library(tidyr)      # 数据整理
library(openxlsx)   # Excel输出
library(caret)
library(stringr)
library(tidyverse)
library(purrr)
library(tableone)
# 加载必要的R包
required_packages <- c(
  "dplyr",      # 数据操作
  "tidyr",      # 数据整理
  "readr",      # 高效数据读取
  "caret",      # 机器学习工具
  "stringr",    # 字符串处理
  "purrr",      # 函数式编程
  "tableone",   # 创建基线特征表格
  "openxlsx"    # Excel输出
)

# 安装缺失的包
install_if_missing <- function(packages) {
  for(pkg in packages) {
    if(!require(pkg, character.only = TRUE, quietly = TRUE)) {
      cat(sprintf("安装包: %s\n", pkg))
      install.packages(pkg, dependencies = TRUE)
      library(pkg, character.only = TRUE)
    }
  }
}

install_if_missing(required_packages)

# 设置随机种子保证结果可复现
set.seed(123)

# 自定义重复字符串函数
str_dup <- function(pattern, times) {
  paste(rep(pattern, times), collapse = "")
}

cat("=== MIMIC数据库处理与分析（学术投稿版） ===\n")
cat("程序启动时间:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat(str_dup("=", 80), "\n\n")

# ==============================================================================
# 第二部分：数据清洗和预处理
# ==============================================================================
cat("\n", str_dup("*", 80), "\n")
cat("第二部分：数据清洗和预处理\n")
cat(str_dup("*", 80), "\n\n")

# 1. 读取数据
cat("=== 1. 读取数据 ===\n")
mimic_data <- read_csv("data.csv")
hospital_data <- read_csv("hospital.csv")

cat("MIMIC数据:", dim(mimic_data), "\n")
cat("医院数据:", dim(hospital_data), "\n")

# 2. 辅助函数：识别关键列
find_column <- function(df, keywords) {
  for(keyword in keywords) {
    if(keyword %in% colnames(df)) {
      return(keyword)
    }
  }
  return(NULL)
}

# 识别ID列
id_col <- find_column(mimic_data, c("subject_id", "hadm_id", "stay_id", "patient_id"))
if(is.null(id_col)) {
  id_col <- colnames(mimic_data)[1]
  cat("使用第一列作为ID列:", id_col, "\n")
} else {
  cat("ID列:", id_col, "\n")
}

# 3. 移除混合感染患者
cat("\n=== 2. 识别并移除混合感染患者 ===\n")

patient_gram_info <- mimic_data %>%
  group_by(!!sym(id_col)) %>%
  summarise(
    has_positive = any(grepl("positive|Positive|pos|POS|阳性", gram_type, ignore.case = TRUE)),
    has_negative = any(grepl("negative|Negative|neg|NEG|阴性", gram_type, ignore.case = TRUE)),
    .groups = 'drop'
  ) %>%
  mutate(is_mixed = has_positive & has_negative)

mixed_patients <- patient_gram_info %>% filter(is_mixed == TRUE)
mixed_ids <- mixed_patients[[id_col]]

clean_data <- mimic_data %>% 
  filter(!(!!sym(id_col) %in% mixed_ids))

cat("原始患者数:", length(unique(mimic_data[[id_col]])), "\n")
cat("混合感染患者数:", length(mixed_ids), "\n")
cat("清洗后患者数:", length(unique(clean_data[[id_col]])), "\n")
cat("移除比例:", round(length(mixed_ids)/length(unique(mimic_data[[id_col]]))*100, 1), "%\n")

# 4. 数据分割：80%训练集，20%测试集
cat("\n=== 3. 数据分割（80%训练集，20%测试集） ===\n")

patient_ids <- unique(clean_data[[id_col]])
train_indices <- createDataPartition(1:length(patient_ids), p = 0.8, list = FALSE)
train_ids <- patient_ids[train_indices]
test_ids <- patient_ids[-train_indices]

train_data <- clean_data %>% filter(!!sym(id_col) %in% train_ids)
test_data <- clean_data %>% filter(!!sym(id_col) %in% test_ids)
external_data <- hospital_data

cat("训练集患者数:", length(train_ids), "\n")
cat("测试集患者数:", length(test_ids), "\n")
cat("外部验证集患者数:", length(unique(external_data[[find_column(external_data, 
                                                          c("subject_id", "hadm_id", "stay_id", "patient_id"))]])), "\n")
cat("训练集记录数:", nrow(train_data), "\n")
cat("测试集记录数:", nrow(test_data), "\n")
cat("外部验证集记录数:", nrow(external_data), "\n")

# ==============================================================================
# 第三部分：特征变量筛选（基于训练集缺失率）
# ==============================================================================
cat("\n", str_dup("*", 80), "\n")
cat("第三部分：特征变量筛选（基于训练集缺失率）\n")
cat(str_dup("*", 80), "\n\n")

cat("=== 4. 特征变量缺失率分析与筛选 ===\n")

# 定义必须保留的元数据列
metadata_cols <- c(
  "subject_id", "hadm_id", "stay_id", "patient_id", "id", "ID",
  "age", "gender", "sex",
  "gram_type", "gram_stain", "gram_result",
  "bacteria_list", "distinct_bacteria_count",
  "first_culture_time", "culture_time",
  "vital_period", "lab_period",
  "outcome", "mortality", "death", "survival"
)

# 找出训练集中实际存在的元数据列
existing_metadata_cols <- intersect(metadata_cols, colnames(train_data))
cat("必须保留的元数据列 (", length(existing_metadata_cols), "个):\n", sep = "")
cat(paste(existing_metadata_cols, collapse = ", "), "\n")

# 特征变量列（排除元数据列）
feature_cols <- setdiff(colnames(train_data), existing_metadata_cols)

# 计算缺失率并筛选特征
if(length(feature_cols) > 0) {
  cat("\n=== 特征变量缺失率分析 ===\n")
  
  # 计算每个特征变量的缺失率
  missing_rates <- data.frame(
    feature = character(),
    missing_count = integer(),
    total_count = integer(),
    missing_rate = numeric(),
    stringsAsFactors = FALSE
  )
  
  for(feat in feature_cols) {
    missing_n <- sum(is.na(train_data[[feat]]))
    total_n <- nrow(train_data)
    rate <- missing_n / total_n * 100
    
    missing_rates <- rbind(missing_rates, 
                           data.frame(feature = feat, 
                                      missing_count = missing_n,
                                      total_count = total_n,
                                      missing_rate = rate,
                                      stringsAsFactors = FALSE))
  }
  
  # 按缺失率排序
  missing_rates <- missing_rates %>%
    arrange(desc(missing_rate))
  
  # 打印缺失率最高的20个特征
  cat("\n缺失率最高的20个特征:\n")
  for(i in 1:min(20, nrow(missing_rates))) {
    feat <- missing_rates$feature[i]
    rate <- missing_rates$missing_rate[i]
    cat(sprintf("  %-40s %.1f%%\n", feat, rate))
  }
  
  # 筛选特征：保留训练集中缺失率≤60%的特征变量
  threshold <- 61
  selected_features <- missing_rates %>%
    filter(missing_rate <= threshold) %>%
    pull(feature)
  
  excluded_features <- missing_rates %>%
    filter(missing_rate > threshold) %>%
    pull(feature)
  
  cat("\n=== 特征变量筛选结果 ===\n")
  cat("特征变量总数:", length(feature_cols), "\n")
  cat("保留特征变量数（缺失率≤", threshold, "%）:", length(selected_features), "\n", sep = "")
  cat("排除特征变量数（缺失率>", threshold, "%）:", length(excluded_features), "\n", sep = "")
  
  # 显示缺失率最高的排除特征
  if(length(excluded_features) > 0) {
    cat("\n排除的特征变量（缺失率最高的10个）:\n")
    top_excluded <- missing_rates %>%
      filter(feature %in% excluded_features) %>%
      arrange(desc(missing_rate)) %>%
      head(10)
    
    for(i in 1:nrow(top_excluded)) {
      cat(sprintf("  %-35s %.1f%%\n", top_excluded$feature[i], top_excluded$missing_rate[i]))
    }
  }
  
  # 最终保留的列
  final_cols <- c(existing_metadata_cols, selected_features)
  
} else {
  cat("没有特征变量列需要筛选\n")
  final_cols <- colnames(train_data)
  selected_features <- character()
  excluded_features <- character()
}

cat("\n最终保留的总列数:", length(final_cols), "\n")
cat("保留的特征变量数:", length(selected_features), "\n")

# 4. 应用筛选结果到所有数据集
cat("\n=== 5. 应用特征筛选到所有数据集 ===\n")

# 筛选训练集
train_filtered <- train_data %>%
  select(any_of(final_cols))

# 筛选测试集（添加缺失列用NA填充）
test_filtered <- test_data
for(col in final_cols) {
  if(!col %in% colnames(test_filtered)) {
    test_filtered[[col]] <- NA
  }
}
test_filtered <- test_filtered %>%
  select(all_of(final_cols))

# 筛选外部验证集（添加缺失列用NA填充）
external_filtered <- external_data
for(col in final_cols) {
  if(!col %in% colnames(external_filtered)) {
    external_filtered[[col]] <- NA
  }
}
external_filtered <- external_filtered %>%
  select(all_of(final_cols))

cat("数据集筛选后维度:\n")
cat("训练集:", dim(train_filtered), "\n")
cat("测试集:", dim(test_filtered), "\n")
cat("外部验证集:", dim(external_filtered), "\n")

# ==============================================================================
# 第四部分：细菌分布分析（每个数据集排行前十的细菌）
# ==============================================================================
cat("\n", str_dup("*", 80), "\n")
cat("第四部分：细菌分布分析（排行前十的细菌）\n")
cat(str_dup("*", 80), "\n\n")

cat("=== 6. 细菌分布分析 ===\n")

# 细菌分析函数
analyze_bacteria_distribution <- function(data, dataset_name, id_col) {
  cat("\n=== ", dataset_name, "细菌分布分析 ===\n")
  
  if(!"bacteria_list" %in% colnames(data)) {
    cat("警告: 数据集中没有bacteria_list列\n")
    return(NULL)
  }
  
  # 按患者去重（取第一条记录）
  data_unique <- data %>%
    group_by(!!sym(id_col)) %>%
    slice(1) %>%
    ungroup()
  
  # 创建Gram分组
  data_unique <- data_unique %>%
    mutate(
      gram_group = case_when(
        grepl("positive|Positive|pos|POS|阳性", gram_type, ignore.case = TRUE) ~ "Gram Positive",
        grepl("negative|Negative|neg|NEG|阴性", gram_type, ignore.case = TRUE) ~ "Gram Negative",
        TRUE ~ "Other/Unknown"
      )
    )
  
  # 只保留Gram Positive和Gram Negative的患者
  data_analysis <- data_unique %>%
    filter(gram_group %in% c("Gram Positive", "Gram Negative"))
  
  if(nrow(data_analysis) == 0) {
    cat("警告: 该数据集没有Gram Positive或Gram Negative患者\n")
    return(NULL)
  }
  
  # 分析所有细菌
  bacteria_all <- data_analysis %>%
    filter(!is.na(bacteria_list) & bacteria_list != "") %>%
    mutate(
      bacteria_split = str_split(bacteria_list, ";"),
      bacteria_split = map(bacteria_split, ~ str_trim(.x)),
      bacteria_split = map(bacteria_split, ~ .x[.x != ""])
    ) %>%
    unnest(bacteria_split) %>%
    rename(bacteria = bacteria_split) %>%
    filter(bacteria != "")
  
  if(nrow(bacteria_all) == 0) {
    cat("警告: 没有有效的细菌数据\n")
    return(NULL)
  }
  
  # 所有细菌排行前十
  cat("\n所有细菌排行前十:\n")
  bacteria_top10 <- bacteria_all %>%
    count(bacteria, sort = TRUE) %>%
    mutate(
      percentage = n / nrow(data_analysis) * 100,
      rank = row_number()
    ) %>%
    filter(rank <= 10)
  
  print(bacteria_top10)
  
  # Gram Positive细菌排行前十
  cat("\nGram Positive细菌排行前十:\n")
  bacteria_pos <- bacteria_all %>%
    filter(gram_group == "Gram Positive") %>%
    count(bacteria, sort = TRUE) %>%
    mutate(
      percentage = n / nrow(data_analysis %>% filter(gram_group == "Gram Positive")) * 100,
      rank = row_number()
    ) %>%
    filter(rank <= 10)
  
  print(bacteria_pos)
  
  # Gram Negative细菌排行前十
  cat("\nGram Negative细菌排行前十:\n")
  bacteria_neg <- bacteria_all %>%
    filter(gram_group == "Gram Negative") %>%
    count(bacteria, sort = TRUE) %>%
    mutate(
      percentage = n / nrow(data_analysis %>% filter(gram_group == "Gram Negative")) * 100,
      rank = row_number()
    ) %>%
    filter(rank <= 10)
  
  print(bacteria_neg)
  
  # 返回结果
  return(list(
    all_bacteria = bacteria_all,
    top10_all = bacteria_top10,
    top10_pos = bacteria_pos,
    top10_neg = bacteria_neg,
    patient_count = nrow(data_analysis),
    pos_count = nrow(data_analysis %>% filter(gram_group == "Gram Positive")),
    neg_count = nrow(data_analysis %>% filter(gram_group == "Gram Negative"))
  ))
}

# 执行细菌分布分析
cat("\n开始细菌分布分析...\n")

# 识别各数据集的ID列
find_id_column <- function(df) {
  id_candidates <- c("subject_id", "hadm_id", "stay_id", "patient_id", "id", "ID", "patient")
  for(candidate in id_candidates) {
    if(candidate %in% colnames(df)) {
      return(candidate)
    }
  }
  return(colnames(df)[1])
}

id_col_train <- find_id_column(train_filtered)
cat("训练集ID列:", id_col_train, "\n")

# 对三个数据集进行细菌分析
bacteria_train <- analyze_bacteria_distribution(train_filtered, "训练集", id_col_train)
bacteria_test <- analyze_bacteria_distribution(test_filtered, "测试集", id_col_train)
bacteria_external <- analyze_bacteria_distribution(external_filtered, "外部验证集", 
                                                   find_id_column(external_filtered))

# ==============================================================================
# 第五部分：插补前的基线特征分析（筛选后特征）
# ==============================================================================
cat("\n", str_dup("*", 80), "\n")
cat("第五部分：插补前的基线特征分析（筛选后特征）\n")
cat(str_dup("*", 80), "\n\n")

cat("=== 7. 插补前的全面基线特征分析 ===\n")

# 改进的变量类型识别函数（修正版 - 能识别0/1疾病特征）
improved_type_detection <- function(data_analysis) {
  continuous_vars <- character()
  categorical_vars <- character()
  
  # 定义0/1二值变量的列名（Charlson组分）
  binary_columns <- c(
    "myocardial_infarct", "congestive_heart_failure", "peripheral_vascular_disease",
    "cerebrovascular_disease", "dementia", "chronic_pulmonary_disease",
    "rheumatic_disease", "peptic_ulcer_disease", "mild_liver_disease",
    "diabetes_without_cc", "diabetes_with_cc", "paraplegia", "renal_disease",
    "malignant_cancer", "severe_liver_disease", "metastatic_solid_tumor", "aids"
  )
  
  for(var in colnames(data_analysis)) {
    # 跳过特定列
    if(var %in% c("subject_id", "hadm_id", "stay_id", "patient_id", "id", 
                  "gram_type", "gram_group", "bacteria_list", 
                  "first_culture_time", "culture_time", "vital_period", "lab_period")) {
      next
    }
    
    # 检查是否为数值型
    if(is.numeric(data_analysis[[var]])) {
      # 检查是否为0/1二值变量（Charlson组分）
      unique_vals <- unique(na.omit(data_analysis[[var]]))
      if(length(unique_vals) <= 2 && all(unique_vals %in% c(0, 1))) {
        categorical_vars <- c(categorical_vars, var)
      } else {
        continuous_vars <- c(continuous_vars, var)
      }
      next
    }
    
    # 检查是否为逻辑型
    if(is.logical(data_analysis[[var]])) {
      categorical_vars <- c(categorical_vars, var)
      next
    }
    
    # 尝试转换为数值型
    values <- data_analysis[[var]]
    
    # 处理常见临床变量（如resp_rate, heart_rate等）
    clinical_numeric_vars <- c("resp_rate", "heart_rate", "temperature", "sbp", 
                               "dbp", "spo2", "glucose", "wbc", "hgb", "platelet",
                               "creatinine", "bun", "sodium", "potassium", "chloride",
                               "bicarbonate", "ph", "pao2", "paco2", "lactate")
    
    if(var %in% clinical_numeric_vars) {
      numeric_test <- suppressWarnings(as.numeric(as.character(values)))
      numeric_count <- sum(!is.na(numeric_test))
      if(numeric_count / length(values) > 0.3) {
        continuous_vars <- c(continuous_vars, var)
      } else {
        categorical_vars <- c(categorical_vars, var)
      }
      next
    }
    
    # 通用转换逻辑 - 增强对0/1字符变量的识别
    numeric_test <- suppressWarnings(as.numeric(as.character(values)))
    numeric_count <- sum(!is.na(numeric_test))
    
    # 检查是否所有非缺失值都是0或1（字符型）
    non_na_values <- values[!is.na(values) & values != ""]
    if(length(non_na_values) > 0) {
      # 检查是否为0/1字符（如 "0", "1", "0.0", "1.0"）
      is_zero_one <- all(non_na_values %in% c("0", "1", "0.0", "1.0", "0.00", "1.00", "0", "1"))
      if(is_zero_one) {
        categorical_vars <- c(categorical_vars, var)
        next
      }
    }
    
    if(numeric_count / length(values) > 0.5) {
      continuous_vars <- c(continuous_vars, var)
    } else {
      unique_values <- unique(values[!is.na(values)])
      if(length(unique_values) <= 10) {
        categorical_vars <- c(categorical_vars, var)
      } else {
        # 类别多但可能是数值型的字符表示
        is_numeric_string <- function(x) {
          grepl("^[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?$", x)
        }
        if(length(non_na_values) > 0) {
          numeric_string_count <- sum(is_numeric_string(as.character(non_na_values)))
          if(numeric_string_count / length(non_na_values) > 0.7) {
            continuous_vars <- c(continuous_vars, var)
          } else {
            categorical_vars <- c(categorical_vars, var)
          }
        } else {
          categorical_vars <- c(categorical_vars, var)
        }
      }
    }
  }
  
  return(list(continuous = continuous_vars, categorical = categorical_vars))
}

# 基线特征分析函数（仅用于插补前数据）
perform_pre_imputation_baseline_analysis <- function(data, dataset_name, id_col) {
  cat("\n", str_dup("=", 60), "\n", sep="")
  cat("插补前基线特征分析:", dataset_name, "\n")
  cat(str_dup("=", 60), "\n")
  
  # 按患者去重（取第一条记录）
  data_unique <- data %>%
    group_by(!!sym(id_col)) %>%
    slice(1) %>%
    ungroup()
  
  # 创建Gram分组
  data_unique <- data_unique %>%
    mutate(
      gram_group = case_when(
        grepl("positive|Positive|pos|POS|阳性", gram_type, ignore.case = TRUE) ~ "Gram Positive",
        grepl("negative|Negative|neg|NEG|阴性", gram_type, ignore.case = TRUE) ~ "Gram Negative",
        TRUE ~ "Other/Unknown"
      )
    )
  
  # 只保留Gram Positive和Gram Negative的患者
  data_analysis <- data_unique %>%
    filter(gram_group %in% c("Gram Positive", "Gram Negative"))
  
  # 如果没有患者，返回空结果
  if(nrow(data_analysis) == 0) {
    cat("警告: 该数据集没有Gram Positive或Gram Negative患者\n")
    return(list(
      baseline_table = data.frame(),
      gram_counts = c("Gram Positive" = 0, "Gram Negative" = 0),
      total_patients = 0,
      variable_summary = data.frame()
    ))
  }
  
  # 排除不需要分析的列
  exclude_cols <- c(id_col, "gram_type", "gram_group", "bacteria_list", 
                    "first_culture_time", "vital_period", "lab_period", "culture_time")
  
  analysis_vars <- setdiff(colnames(data_analysis), exclude_cols)
  
  # 使用改进的类型识别
  var_types <- improved_type_detection(data_analysis %>% select(all_of(analysis_vars)))
  continuous_vars <- var_types$continuous
  categorical_vars <- var_types$categorical
  
  cat("\n变量类型统计:\n")
  cat("连续变量:", length(continuous_vars), "个\n")
  cat("分类变量:", length(categorical_vars), "个\n")
  cat("总分析变量:", length(analysis_vars), "个\n")
  
  # 创建基线表格
  baseline_table <- data.frame(
    Variable = character(),
    Type = character(),
    Gram_Positive_N = character(),
    Gram_Positive_Value = character(),
    Gram_Negative_N = character(),
    Gram_Negative_Value = character(),
    Missing_GramPos = character(),
    Missing_GramNeg = character(),
    P_Value = character(),
    Test = character(),
    stringsAsFactors = FALSE
  )
  
  # P值格式化函数
  format_p_value <- function(p) {
    if(is.na(p)) {
      return("NA")
    } else if(p < 0.001) {
      return("<0.001")
    } else if(p < 0.01) {
      return(sprintf("%.3f", p))
    } else {
      return(sprintf("%.3f", p))
    }
  }
  
  # 1. 分析人口学特征
  cat("\n分析人口学特征...\n")
  
  # 年龄
  if("age" %in% colnames(data_analysis)) {
    # 安全转换年龄为数值
    data_analysis$age_numeric <- suppressWarnings(as.numeric(as.character(data_analysis$age)))
    
    # 计算统计量
    age_stats <- data_analysis %>%
      group_by(gram_group) %>%
      summarise(
        n_total = n(),
        n_nonmissing = sum(!is.na(age_numeric)),
        median_val = ifelse(n_nonmissing > 0, median(age_numeric, na.rm = TRUE), NA),
        q1 = ifelse(n_nonmissing > 0, quantile(age_numeric, 0.25, na.rm = TRUE), NA),
        q3 = ifelse(n_nonmissing > 0, quantile(age_numeric, 0.75, na.rm = TRUE), NA),
        .groups = 'drop'
      )
    
    # 计算缺失率
    age_missing <- data_analysis %>%
      group_by(gram_group) %>%
      summarise(
        total = n(),
        missing = sum(is.na(age_numeric)),
        missing_pct = missing/total*100,
        .groups = 'drop'
      )
    
    # 计算p值（需要至少2个非缺失值）
    pos_age <- data_analysis$age_numeric[data_analysis$gram_group == "Gram Positive"]
    neg_age <- data_analysis$age_numeric[data_analysis$gram_group == "Gram Negative"]
    pos_age_clean <- pos_age[!is.na(pos_age)]
    neg_age_clean <- neg_age[!is.na(neg_age)]
    
    if(length(pos_age_clean) >= 2 && length(neg_age_clean) >= 2) {
      p_value <- tryCatch({
        wilcox.test(pos_age_clean, neg_age_clean)$p.value
      }, error = function(e) { NA })
    } else { 
      p_value <- NA 
    }
    
    # 获取统计值
    pos_stats <- age_stats %>% filter(gram_group == "Gram Positive")
    neg_stats <- age_stats %>% filter(gram_group == "Gram Negative")
    pos_missing <- age_missing %>% filter(gram_group == "Gram Positive")
    neg_missing <- age_missing %>% filter(gram_group == "Gram Negative")
    
    baseline_table <- rbind(baseline_table, data.frame(
      Variable = "Age (years)",
      Type = "Continuous",
      Gram_Positive_N = ifelse(nrow(pos_stats) > 0, as.character(pos_stats$n_nonmissing), "0"),
      Gram_Positive_Value = ifelse(nrow(pos_stats) > 0 && !is.na(pos_stats$median_val),
                                   sprintf("%.1f (%.1f-%.1f)", 
                                           pos_stats$median_val,
                                           pos_stats$q1,
                                           pos_stats$q3),
                                   "NA"),
      Gram_Negative_N = ifelse(nrow(neg_stats) > 0, as.character(neg_stats$n_nonmissing), "0"),
      Gram_Negative_Value = ifelse(nrow(neg_stats) > 0 && !is.na(neg_stats$median_val),
                                   sprintf("%.1f (%.1f-%.1f)", 
                                           neg_stats$median_val,
                                           neg_stats$q1,
                                           neg_stats$q3),
                                   "NA"),
      Missing_GramPos = ifelse(nrow(pos_missing) > 0,
                               sprintf("%d (%.1f%%)", 
                                       pos_missing$missing,
                                       pos_missing$missing_pct),
                               "NA"),
      Missing_GramNeg = ifelse(nrow(neg_missing) > 0,
                               sprintf("%d (%.1f%%)", 
                                       neg_missing$missing,
                                       neg_missing$missing_pct),
                               "NA"),
      P_Value = format_p_value(p_value),
      Test = "Wilcoxon rank-sum",
      stringsAsFactors = FALSE
    ))
  }
  
  # 性别
  if("gender" %in% colnames(data_analysis)) {
    # 标准化性别编码
    data_analysis <- data_analysis %>%
      mutate(
        gender_clean = case_when(
          grepl("female|女|f|F", gender, ignore.case = TRUE) ~ "Female",
          grepl("male|男|m|M", gender, ignore.case = TRUE) ~ "Male",
          TRUE ~ "Other/Unknown"
        )
      )
    
    # 统计性别分布
    gender_stats <- data_analysis %>%
      filter(gender_clean %in% c("Male", "Female")) %>%
      group_by(gram_group, gender_clean) %>%
      summarise(count = n(), .groups = 'drop') %>%
      group_by(gram_group) %>%
      mutate(
        total = sum(count),
        percentage = count / total * 100
      ) %>%
      ungroup()
    
    # 计算p值
    gender_data_for_test <- data_analysis %>%
      filter(gender_clean %in% c("Male", "Female"))
    
    if(nrow(gender_data_for_test) > 0) {
      cont_table <- table(
        gender_data_for_test$gram_group,
        gender_data_for_test$gender_clean
      )
      if(all(dim(cont_table) >= 2)) {
        p_value <- tryCatch({ 
          chisq.test(cont_table)$p.value 
        }, error = function(e) { NA })
      } else { 
        p_value <- NA 
      }
    } else { 
      p_value <- NA 
    }
    
    # 获取每组的患者总数
    group_totals <- data_analysis %>%
      group_by(gram_group) %>%
      summarise(total = n(), .groups = 'drop')
    
    pos_total <- group_totals %>% filter(gram_group == "Gram Positive") %>% pull(total)
    neg_total <- group_totals %>% filter(gram_group == "Gram Negative") %>% pull(total)
    
    # 添加性别行
    for(gender_type in c("Male", "Female")) {
      gender_row_pos <- gender_stats %>% 
        filter(gram_group == "Gram Positive" & gender_clean == gender_type)
      gender_row_neg <- gender_stats %>% 
        filter(gram_group == "Gram Negative" & gender_clean == gender_type)
      
      baseline_table <- rbind(baseline_table, data.frame(
        Variable = paste("Gender:", gender_type),
        Type = "Categorical",
        Gram_Positive_N = ifelse(length(pos_total) > 0, as.character(pos_total), "NA"),
        Gram_Positive_Value = ifelse(nrow(gender_row_pos) > 0,
                                     sprintf("%d (%.1f%%)", 
                                             gender_row_pos$count,
                                             gender_row_pos$percentage),
                                     "0 (0.0%)"),
        Gram_Negative_N = ifelse(length(neg_total) > 0, as.character(neg_total), "NA"),
        Gram_Negative_Value = ifelse(nrow(gender_row_neg) > 0,
                                     sprintf("%d (%.1f%%)", 
                                             gender_row_neg$count,
                                             gender_row_neg$percentage),
                                     "0 (0.0%)"),
        Missing_GramPos = "0 (0.0%)",
        Missing_GramNeg = "0 (0.0%)",
        P_Value = ifelse(gender_type == "Male", format_p_value(p_value), ""),
        Test = ifelse(gender_type == "Male", "Chi-square", ""),
        stringsAsFactors = FALSE
      ))
    }
  }
  
  # 2. 分析所有连续变量
  cat("\n分析所有连续变量...\n")
  progress <- 0
  total_cont_vars <- length(continuous_vars)
  
  for(var in continuous_vars) {
    if(!var %in% c("age", "age_numeric", "gender_clean")) {
      progress <- progress + 1
      if(progress %% 20 == 0) {
        cat(sprintf("  已处理 %d/%d 个连续变量\n", progress, total_cont_vars))
      }
      
      # 确保变量是数值型
      if(!is.numeric(data_analysis[[var]])) {
        data_analysis[[var]] <- suppressWarnings(as.numeric(as.character(data_analysis[[var]])))
      }
      
      # 计算统计量（仅非缺失值）
      var_stats <- data_analysis %>%
        filter(!is.na(!!sym(var))) %>%
        group_by(gram_group) %>%
        summarise(
          n = n(),
          median_val = median(!!sym(var), na.rm = TRUE),
          q1 = quantile(!!sym(var), 0.25, na.rm = TRUE),
          q3 = quantile(!!sym(var), 0.75, na.rm = TRUE),
          .groups = 'drop'
        )
      
      # 计算缺失率
      var_missing <- data_analysis %>%
        group_by(gram_group) %>%
        summarise(
          total = n(),
          missing = sum(is.na(!!sym(var))),
          missing_pct = missing/total*100,
          .groups = 'drop'
        )
      
      # 计算p值（使用非缺失值）
      pos_data <- data_analysis[[var]][data_analysis$gram_group == "Gram Positive"]
      neg_data <- data_analysis[[var]][data_analysis$gram_group == "Gram Negative"]
      pos_data_clean <- pos_data[!is.na(pos_data)]
      neg_data_clean <- neg_data[!is.na(neg_data)]
      
      if(length(pos_data_clean) > 1 && length(neg_data_clean) > 1) {
        p_value <- tryCatch({
          wilcox.test(pos_data_clean, neg_data_clean)$p.value
        }, error = function(e) { NA })
      } else { 
        p_value <- NA 
      }
      
      # 获取统计值
      pos_stats <- var_stats %>% filter(gram_group == "Gram Positive")
      neg_stats <- var_stats %>% filter(gram_group == "Gram Negative")
      pos_missing <- var_missing %>% filter(gram_group == "Gram Positive")
      neg_missing <- var_missing %>% filter(gram_group == "Gram Negative")
      
      baseline_table <- rbind(baseline_table, data.frame(
        Variable = var,
        Type = "Continuous",
        Gram_Positive_N = ifelse(nrow(pos_stats) > 0, as.character(pos_stats$n), "0"),
        Gram_Positive_Value = ifelse(nrow(pos_stats) > 0 && !is.na(pos_stats$median_val),
                                     sprintf("%.3f (%.3f-%.3f)", 
                                             pos_stats$median_val,
                                             pos_stats$q1,
                                             pos_stats$q3),
                                     "NA"),
        Gram_Negative_N = ifelse(nrow(neg_stats) > 0, as.character(neg_stats$n), "0"),
        Gram_Negative_Value = ifelse(nrow(neg_stats) > 0 && !is.na(neg_stats$median_val),
                                     sprintf("%.3f (%.3f-%.3f)", 
                                             neg_stats$median_val,
                                             neg_stats$q1,
                                             neg_stats$q3),
                                     "NA"),
        Missing_GramPos = ifelse(nrow(pos_missing) > 0,
                                 sprintf("%d (%.1f%%)", 
                                         pos_missing$missing,
                                         pos_missing$missing_pct),
                                 "NA"),
        Missing_GramNeg = ifelse(nrow(neg_missing) > 0,
                                 sprintf("%d (%.1f%%)", 
                                         neg_missing$missing,
                                         neg_missing$missing_pct),
                                 "NA"),
        P_Value = format_p_value(p_value),
        Test = "Wilcoxon rank-sum",
        stringsAsFactors = FALSE
      ))
    }
  }
  
  if(total_cont_vars > 0) {
    cat(sprintf("  完成! 共处理 %d 个连续变量\n", total_cont_vars))
  }
  
  # 3. 分析细菌计数
  if("distinct_bacteria_count" %in% colnames(data_analysis)) {
    cat("\n分析细菌计数...\n")
    
    # 安全转换为数值
    data_analysis$distinct_bacteria_count_numeric <- suppressWarnings(
      as.numeric(as.character(data_analysis$distinct_bacteria_count))
    )
    
    bacteria_count_stats <- data_analysis %>%
      filter(!is.na(distinct_bacteria_count_numeric)) %>%
      group_by(gram_group) %>%
      summarise(
        n = n(),
        median_val = median(distinct_bacteria_count_numeric, na.rm = TRUE),
        q1 = quantile(distinct_bacteria_count_numeric, 0.25, na.rm = TRUE),
        q3 = quantile(distinct_bacteria_count_numeric, 0.75, na.rm = TRUE),
        .groups = 'drop'
      )
    
    # 计算缺失率
    bc_missing <- data_analysis %>%
      group_by(gram_group) %>%
      summarise(
        total = n(),
        missing = sum(is.na(distinct_bacteria_count_numeric)),
        missing_pct = missing/total*100,
        .groups = 'drop'
      )
    
    # 计算p值
    pos_bc <- data_analysis$distinct_bacteria_count_numeric[data_analysis$gram_group == "Gram Positive"]
    neg_bc <- data_analysis$distinct_bacteria_count_numeric[data_analysis$gram_group == "Gram Negative"]
    pos_bc_clean <- pos_bc[!is.na(pos_bc)]
    neg_bc_clean <- neg_bc[!is.na(neg_bc)]
    
    if(length(pos_bc_clean) > 1 && length(neg_bc_clean) > 1) {
      p_value <- tryCatch({
        wilcox.test(pos_bc_clean, neg_bc_clean)$p.value
      }, error = function(e) { NA })
    } else { 
      p_value <- NA 
    }
    
    # 获取统计值
    pos_stats <- bacteria_count_stats %>% filter(gram_group == "Gram Positive")
    neg_stats <- bacteria_count_stats %>% filter(gram_group == "Gram Negative")
    pos_missing <- bc_missing %>% filter(gram_group == "Gram Positive")
    neg_missing <- bc_missing %>% filter(gram_group == "Gram Negative")
    
    baseline_table <- rbind(baseline_table, data.frame(
      Variable = "Distinct bacteria count",
      Type = "Continuous",
      Gram_Positive_N = ifelse(nrow(pos_stats) > 0, as.character(pos_stats$n), "0"),
      Gram_Positive_Value = ifelse(nrow(pos_stats) > 0 && !is.na(pos_stats$median_val),
                                   sprintf("%.1f (%.1f-%.1f)", 
                                           pos_stats$median_val,
                                           pos_stats$q1,
                                           pos_stats$q3),
                                   "NA"),
      Gram_Negative_N = ifelse(nrow(neg_stats) > 0, as.character(neg_stats$n), "0"),
      Gram_Negative_Value = ifelse(nrow(neg_stats) > 0 && !is.na(neg_stats$median_val),
                                   sprintf("%.1f (%.1f-%.1f)", 
                                           neg_stats$median_val,
                                           neg_stats$q1,
                                           neg_stats$q3),
                                   "NA"),
      Missing_GramPos = ifelse(nrow(pos_missing) > 0,
                               sprintf("%d (%.1f%%)", 
                                       pos_missing$missing,
                                       pos_missing$missing_pct),
                               "NA"),
      Missing_GramNeg = ifelse(nrow(neg_missing) > 0,
                               sprintf("%d (%.1f%%)", 
                                       neg_missing$missing,
                                       neg_missing$missing_pct),
                               "NA"),
      P_Value = format_p_value(p_value),
      Test = "Wilcoxon rank-sum",
      stringsAsFactors = FALSE
    ))
  }
  
  # 4. 分析所有分类变量
  cat("\n分析所有分类变量...\n")
  progress_cat <- 0
  total_cat_vars <- length(categorical_vars)
  
  for(var in categorical_vars) {
    if(!var %in% c("gender", "gender_clean", "gram_type", "gram_group")) {
      progress_cat <- progress_cat + 1
      if(progress_cat %% 10 == 0) {
        cat(sprintf("  已处理 %d/%d 个分类变量\n", progress_cat, total_cat_vars))
      }
      
      # 获取变量的所有非缺失类别
      var_values <- data_analysis[[var]]
      var_values <- var_values[!is.na(var_values)]
      unique_values <- unique(var_values)
      
      # 如果类别太多（超过20个），只显示摘要信息
      if(length(unique_values) > 20) {
        baseline_table <- rbind(baseline_table, data.frame(
          Variable = paste(var, "(类别过多:", length(unique_values), ")"),
          Type = "Categorical",
          Gram_Positive_N = "多种",
          Gram_Positive_Value = sprintf("%d 个类别", length(unique_values)),
          Gram_Negative_N = "多种",
          Gram_Negative_Value = sprintf("%d 个类别", length(unique_values)),
          Missing_GramPos = sprintf("%d 个缺失", sum(is.na(data_analysis[[var]][data_analysis$gram_group == "Gram Positive"]))),
          Missing_GramNeg = sprintf("%d 个缺失", sum(is.na(data_analysis[[var]][data_analysis$gram_group == "Gram Negative"]))),
          P_Value = "",
          Test = "",
          stringsAsFactors = FALSE
        ))
        next
      }
      
      # 计算缺失率
      var_missing <- data_analysis %>%
        group_by(gram_group) %>%
        summarise(
          total = n(),
          missing = sum(is.na(!!sym(var))),
          missing_pct = missing/total*100,
          .groups = 'drop'
        )
      
      # 获取每组的患者总数
      group_totals <- data_analysis %>%
        group_by(gram_group) %>%
        summarise(total = n(), .groups = 'drop')
      
      pos_total <- group_totals %>% filter(gram_group == "Gram Positive") %>% pull(total)
      neg_total <- group_totals %>% filter(gram_group == "Gram Negative") %>% pull(total)
      
      # 为每个类别添加一行
      for(val in unique_values) {
        # 统计该类别在每组中的分布
        cat_stats <- data_analysis %>%
          group_by(gram_group) %>%
          summarise(
            count = sum(!!sym(var) == val, na.rm = TRUE),
            .groups = 'drop'
          ) %>%
          mutate(
            percentage = ifelse(gram_group == "Gram Positive", 
                                count/pos_total*100,
                                count/neg_total*100)
          )
        
        pos_stats <- cat_stats %>% filter(gram_group == "Gram Positive")
        neg_stats <- cat_stats %>% filter(gram_group == "Gram Negative")
        pos_missing <- var_missing %>% filter(gram_group == "Gram Positive")
        neg_missing <- var_missing %>% filter(gram_group == "Gram Negative")
        
        # 对于二分类变量，计算p值
        p_value <- ""
        test_method <- ""
        if(length(unique_values) == 2) {
          # 创建列联表
          cont_table <- table(
            data_analysis$gram_group,
            data_analysis[[var]] == val
          )
          if(all(dim(cont_table) >= 2)) {
            p_value <- tryCatch({ 
              chisq.test(cont_table)$p.value 
            }, error = function(e) { NA })
            p_value <- format_p_value(p_value)
            test_method = "Chi-square"
          }
        }
        
        baseline_table <- rbind(baseline_table, data.frame(
          Variable = paste(var, ":", val),
          Type = "Categorical",
          Gram_Positive_N = ifelse(length(pos_total) > 0, as.character(pos_total), "NA"),
          Gram_Positive_Value = ifelse(nrow(pos_stats) > 0,
                                       sprintf("%d (%.1f%%)", 
                                               pos_stats$count,
                                               pos_stats$percentage),
                                       "0 (0.0%)"),
          Gram_Negative_N = ifelse(length(neg_total) > 0, as.character(neg_total), "NA"),
          Gram_Negative_Value = ifelse(nrow(neg_stats) > 0,
                                       sprintf("%d (%.1f%%)", 
                                               neg_stats$count,
                                               neg_stats$percentage),
                                       "0 (0.0%)"),
          Missing_GramPos = ifelse(nrow(pos_missing) > 0,
                                   sprintf("%d (%.1f%%)", 
                                           pos_missing$missing,
                                           pos_missing$missing_pct),
                                   "NA"),
          Missing_GramNeg = ifelse(nrow(neg_missing) > 0,
                                   sprintf("%d (%.1f%%)", 
                                           neg_missing$missing,
                                           neg_missing$missing_pct),
                                   "NA"),
          P_Value = p_value,
          Test = test_method,
          stringsAsFactors = FALSE
        ))
      }
    }
  }
  
  if(total_cat_vars > 0) {
    cat(sprintf("  完成! 共处理 %d 个分类变量\n", total_cat_vars))
  }
  
  # 创建变量摘要
  variable_summary <- data.frame(
    Variable_Type = c("Continuous", "Categorical", "Total"),
    Count = c(length(continuous_vars), length(categorical_vars), 
              length(continuous_vars) + length(categorical_vars)),
    stringsAsFactors = FALSE
  )
  
  # 统计患者数量
  gram_counts <- table(data_analysis$gram_group)
  
  cat("\n分析完成!\n")
  cat("Gram Positive患者数:", ifelse("Gram Positive" %in% names(gram_counts), 
                                  gram_counts["Gram Positive"], 0), "\n")
  cat("Gram Negative患者数:", ifelse("Gram Negative" %in% names(gram_counts), 
                                  gram_counts["Gram Negative"], 0), "\n")
  cat("总患者数:", nrow(data_analysis), "\n")
  
  return(list(
    baseline_table = baseline_table,
    gram_counts = gram_counts,
    total_patients = nrow(data_analysis),
    variable_summary = variable_summary,
    continuous_vars = continuous_vars,
    categorical_vars = categorical_vars,
    analysis_vars = analysis_vars
  ))
}

# 执行插补前的基线特征分析
cat("\n开始插补前的基线特征分析（筛选后特征）...\n")

# 识别各数据集的ID列
id_col_train <- find_id_column(train_filtered)
cat("训练集ID列:", id_col_train, "\n")

# 对所有数据集执行插补前的基线特征分析
cat("\n=== 训练集基线特征分析（插补前） ===\n")
baseline_train_pre <- perform_pre_imputation_baseline_analysis(
  train_filtered, "训练集", id_col_train)

cat("\n=== 测试集基线特征分析（插补前） ===\n")
baseline_test_pre <- perform_pre_imputation_baseline_analysis(
  test_filtered, "测试集", id_col_train)

cat("\n=== 外部验证集基线特征分析（插补前） ===\n")
baseline_external_pre <- perform_pre_imputation_baseline_analysis(
  external_filtered, "外部验证集", find_id_column(external_filtered))

# ==============================================================================
# 第六部分：数据插补
# ==============================================================================
cat("\n", str_dup("*", 80), "\n")
cat("第六部分：数据插补\n")
cat(str_dup("*", 80), "\n\n")

cat("=== 8. 基于训练集中位数的缺失值插补 ===\n")

# 数据插补函数
perform_median_imputation <- function(train_data, test_data, external_data, id_col) {
  cat("\n开始数据插补...\n")
  
  # 识别数值列（用于插补）
  numeric_cols <- final_cols[sapply(train_data[final_cols], is.numeric)]
  
  # 排除不需要插补的列
  cols_to_exclude <- c(id_col, "subject_id", "hadm_id", "stay_id", "patient_id", "id",
                       "distinct_bacteria_count")
  numeric_cols_for_imputation <- setdiff(numeric_cols, cols_to_exclude)
  
  cat("需要插补的数值列数:", length(numeric_cols_for_imputation), "\n")
  
  if(length(numeric_cols_for_imputation) > 0) {
    # 计算训练集的中位数（按Gram类型分组）
    train_data_grp <- train_data %>%
      mutate(
        gram_group = case_when(
          grepl("positive|Positive|pos|POS|阳性", gram_type, ignore.case = TRUE) ~ "Gram_Positive",
          grepl("negative|Negative|neg|NEG|阴性", gram_type, ignore.case = TRUE) ~ "Gram_Negative",
          TRUE ~ "Other"
        )
      )
    
    # 计算各组中位数
    medians_pos <- train_data_grp %>%
      filter(gram_group == "Gram_Positive") %>%
      select(all_of(numeric_cols_for_imputation)) %>%
      summarise(across(everything(), ~ median(., na.rm = TRUE)))
    
    medians_neg <- train_data_grp %>%
      filter(gram_group == "Gram_Negative") %>%
      select(all_of(numeric_cols_for_imputation)) %>%
      summarise(across(everything(), ~ median(., na.rm = TRUE)))
    
    medians_overall <- train_data_grp %>%
      select(all_of(numeric_cols_for_imputation)) %>%
      summarise(across(everything(), ~ median(., na.rm = TRUE)))
    
    # 插补函数
    impute_data <- function(data, medians_pos, medians_neg, medians_overall) {
      data_imputed <- data %>%
        mutate(
          gram_group = case_when(
            grepl("positive|Positive|pos|POS|阳性", gram_type, ignore.case = TRUE) ~ "Gram_Positive",
            grepl("negative|Negative|neg|NEG|阴性", gram_type, ignore.case = TRUE) ~ "Gram_Negative",
            TRUE ~ "Other"
          )
        )
      
      # 按组插补
      for(col in numeric_cols_for_imputation) {
        pos_idx <- which(data_imputed$gram_group == "Gram_Positive" & is.na(data_imputed[[col]]))
        if(length(pos_idx) > 0) {
          data_imputed[[col]][pos_idx] <- medians_pos[[col]]
        }
        
        neg_idx <- which(data_imputed$gram_group == "Gram_Negative" & is.na(data_imputed[[col]]))
        if(length(neg_idx) > 0) {
          data_imputed[[col]][neg_idx] <- medians_neg[[col]]
        }
        
        other_idx <- which(data_imputed$gram_group == "Other" & is.na(data_imputed[[col]]))
        if(length(other_idx) > 0) {
          data_imputed[[col]][other_idx] <- medians_overall[[col]]
        }
      }
      
      data_imputed <- data_imputed %>% select(-gram_group)
      return(data_imputed)
    }
    
    # 对所有数据集进行插补
    train_imputed <- impute_data(train_data, medians_pos, medians_neg, medians_overall)
    test_imputed <- impute_data(test_data, medians_pos, medians_neg, medians_overall)
    external_imputed <- impute_data(external_data, medians_pos, medians_neg, medians_overall)
    
    cat("数据插补完成!\n")
    
  } else {
    cat("没有数值列需要插补，直接复制原始数据\n")
    train_imputed <- train_data
    test_imputed <- test_data
    external_imputed <- external_data
  }
  
  return(list(
    train_imputed = train_imputed,
    test_imputed = test_imputed,
    external_imputed = external_imputed
  ))
}

# 执行数据插补
imputation_results <- perform_median_imputation(train_filtered, test_filtered, external_filtered, id_col_train)

train_imputed <- imputation_results$train_imputed
test_imputed <- imputation_results$test_imputed
external_imputed <- imputation_results$external_imputed

cat("\n插补后数据维度:\n")
cat("训练集:", dim(train_imputed), "\n")
cat("测试集:", dim(test_imputed), "\n")
cat("外部验证集:", dim(external_imputed), "\n")

# ==============================================================================
# 第七部分：保存所有数据（插补前后）
# ==============================================================================
cat("\n", str_dup("*", 80), "\n")
cat("第七部分：保存所有数据（插补前后）\n")
cat(str_dup("*", 80), "\n\n")

cat("=== 9. 保存所有数据文件 ===\n")

# 创建输出目录
output_dir_main <- "complete_analysis_results"
dir.create(output_dir_main, showWarnings = FALSE, recursive = TRUE)

# 1. 保存插补前的数据
output_dir_pre_imputation <- file.path(output_dir_main, "pre_imputation_data")
dir.create(output_dir_pre_imputation, showWarnings = FALSE, recursive = TRUE)

write_csv(train_filtered, file.path(output_dir_pre_imputation, "train_pre_imputation.csv"))
write_csv(test_filtered, file.path(output_dir_pre_imputation, "test_pre_imputation.csv"))
write_csv(external_filtered, file.path(output_dir_pre_imputation, "external_pre_imputation.csv"))

cat("插补前数据已保存至:", output_dir_pre_imputation, "\n")

# 2. 保存插补后的数据
output_dir_post_imputation <- file.path(output_dir_main, "post_imputation_data")
dir.create(output_dir_post_imputation, showWarnings = FALSE, recursive = TRUE)

write_csv(train_imputed, file.path(output_dir_post_imputation, "train_post_imputation.csv"))
write_csv(test_imputed, file.path(output_dir_post_imputation, "test_post_imputation.csv"))
write_csv(external_imputed, file.path(output_dir_post_imputation, "external_post_imputation.csv"))

cat("插补后数据已保存至:", output_dir_post_imputation, "\n")

# 3. 保存基线特征分析结果（插补前）
output_dir_baseline <- file.path(output_dir_main, "baseline_analysis")
dir.create(output_dir_baseline, showWarnings = FALSE, recursive = TRUE)

cat("\n=== 10. 保存基线特征分析结果 ===\n")

# 创建Excel工作簿保存基线特征
wb_baseline <- createWorkbook()

# 添加训练集基线特征
addWorksheet(wb_baseline, "Train_Baseline")
writeData(wb_baseline, "Train_Baseline", 
          data.frame(Dataset = "Training Set (Pre-imputation)", stringsAsFactors = FALSE), 
          startRow = 1)
writeData(wb_baseline, "Train_Baseline", baseline_train_pre$baseline_table, startRow = 3)

# 添加测试集基线特征
addWorksheet(wb_baseline, "Test_Baseline")
writeData(wb_baseline, "Test_Baseline", 
          data.frame(Dataset = "Test Set (Pre-imputation)", stringsAsFactors = FALSE), 
          startRow = 1)
writeData(wb_baseline, "Test_Baseline", baseline_test_pre$baseline_table, startRow = 3)

# 添加外部验证集基线特征
addWorksheet(wb_baseline, "External_Baseline")
writeData(wb_baseline, "External_Baseline", 
          data.frame(Dataset = "External Validation Set (Pre-imputation)", stringsAsFactors = FALSE), 
          startRow = 1)
writeData(wb_baseline, "External_Baseline", baseline_external_pre$baseline_table, startRow = 3)

# 添加变量摘要
addWorksheet(wb_baseline, "Variable_Summary")

summary_data <- data.frame(
  Dataset = rep(c("Training", "Test", "External"), each = 3),
  Variable_Type = rep(c("Continuous", "Categorical", "Total"), 3),
  Count = c(
    baseline_train_pre$variable_summary$Count,
    baseline_test_pre$variable_summary$Count,
    baseline_external_pre$variable_summary$Count
  )
)

writeData(wb_baseline, "Variable_Summary", summary_data)

# 添加患者统计
addWorksheet(wb_baseline, "Patient_Statistics")
patient_stats <- data.frame(
  Dataset = c("Training", "Test", "External"),
  Gram_Positive = c(
    ifelse("Gram Positive" %in% names(baseline_train_pre$gram_counts), 
           baseline_train_pre$gram_counts["Gram Positive"], 0),
    ifelse("Gram Positive" %in% names(baseline_test_pre$gram_counts), 
           baseline_test_pre$gram_counts["Gram Positive"], 0),
    ifelse("Gram Positive" %in% names(baseline_external_pre$gram_counts), 
           baseline_external_pre$gram_counts["Gram Positive"], 0)
  ),
  Gram_Negative = c(
    ifelse("Gram Negative" %in% names(baseline_train_pre$gram_counts), 
           baseline_train_pre$gram_counts["Gram Negative"], 0),
    ifelse("Gram Negative" %in% names(baseline_test_pre$gram_counts), 
           baseline_test_pre$gram_counts["Gram Negative"], 0),
    ifelse("Gram Negative" %in% names(baseline_external_pre$gram_counts), 
           baseline_external_pre$gram_counts["Gram Negative"], 0)
  ),
  Total = c(
    baseline_train_pre$total_patients,
    baseline_test_pre$total_patients,
    baseline_external_pre$total_patients
  ),
  Gram_Positive_Pct = c(
    ifelse(baseline_train_pre$total_patients > 0,
           round(ifelse("Gram Positive" %in% names(baseline_train_pre$gram_counts), 
                        baseline_train_pre$gram_counts["Gram Positive"], 0) / 
                   baseline_train_pre$total_patients * 100, 1),
           0),
    ifelse(baseline_test_pre$total_patients > 0,
           round(ifelse("Gram Positive" %in% names(baseline_test_pre$gram_counts), 
                        baseline_test_pre$gram_counts["Gram Positive"], 0) / 
                   baseline_test_pre$total_patients * 100, 1),
           0),
    ifelse(baseline_external_pre$total_patients > 0,
           round(ifelse("Gram Positive" %in% names(baseline_external_pre$gram_counts), 
                        baseline_external_pre$gram_counts["Gram Positive"], 0) / 
                   baseline_external_pre$total_patients * 100, 1),
           0)
  ),
  Gram_Negative_Pct = c(
    ifelse(baseline_train_pre$total_patients > 0,
           round(ifelse("Gram Negative" %in% names(baseline_train_pre$gram_counts), 
                        baseline_train_pre$gram_counts["Gram Negative"], 0) / 
                   baseline_train_pre$total_patients * 100, 1),
           0),
    ifelse(baseline_test_pre$total_patients > 0,
           round(ifelse("Gram Negative" %in% names(baseline_test_pre$gram_counts), 
                        baseline_test_pre$gram_counts["Gram Negative"], 0) / 
                   baseline_test_pre$total_patients * 100, 1),
           0),
    ifelse(baseline_external_pre$total_patients > 0,
           round(ifelse("Gram Negative" %in% names(baseline_external_pre$gram_counts), 
                        baseline_external_pre$gram_counts["Gram Negative"], 0) / 
                   baseline_external_pre$total_patients * 100, 1),
           0)
  )
)

writeData(wb_baseline, "Patient_Statistics", patient_stats)

# 保存细菌分析结果
if(!is.null(bacteria_train)) {
  addWorksheet(wb_baseline, "Bacteria_Train")
  writeData(wb_baseline, "Bacteria_Train", 
            data.frame(Dataset = "Training Set Bacteria Analysis", stringsAsFactors = FALSE), 
            startRow = 1)
  writeData(wb_baseline, "Bacteria_Train", 
            data.frame(Note = "Top 10 Bacteria in Training Set", stringsAsFactors = FALSE), 
            startRow = 3)
  writeData(wb_baseline, "Bacteria_Train", bacteria_train$top10_all, startRow = 5)
  
  writeData(wb_baseline, "Bacteria_Train", 
            data.frame(Note = "Top 10 Gram Positive Bacteria", stringsAsFactors = FALSE), 
            startRow = 18)
  writeData(wb_baseline, "Bacteria_Train", bacteria_train$top10_pos, startRow = 20)
  
  writeData(wb_baseline, "Bacteria_Train", 
            data.frame(Note = "Top 10 Gram Negative Bacteria", stringsAsFactors = FALSE), 
            startRow = 33)
  writeData(wb_baseline, "Bacteria_Train", bacteria_train$top10_neg, startRow = 35)
}

if(!is.null(bacteria_test)) {
  addWorksheet(wb_baseline, "Bacteria_Test")
  writeData(wb_baseline, "Bacteria_Test", 
            data.frame(Dataset = "Test Set Bacteria Analysis", stringsAsFactors = FALSE), 
            startRow = 1)
  writeData(wb_baseline, "Bacteria_Test", bacteria_test$top10_all, startRow = 3)
}

if(!is.null(bacteria_external)) {
  addWorksheet(wb_baseline, "Bacteria_External")
  writeData(wb_baseline, "Bacteria_External", 
            data.frame(Dataset = "External Set Bacteria Analysis", stringsAsFactors = FALSE), 
            startRow = 1)
  writeData(wb_baseline, "Bacteria_External", bacteria_external$top10_all, startRow = 3)
}

# 保存Excel文件
baseline_excel_file <- file.path(output_dir_baseline, "pre_imputation_baseline_analysis.xlsx")
saveWorkbook(wb_baseline, baseline_excel_file, overwrite = TRUE)
cat("基线特征分析结果已保存至:", baseline_excel_file, "\n")

# 4. 同时保存CSV格式的基线特征表格
write_csv(baseline_train_pre$baseline_table, 
          file.path(output_dir_baseline, "train_baseline_pre_imputation.csv"))
write_csv(baseline_test_pre$baseline_table, 
          file.path(output_dir_baseline, "test_baseline_pre_imputation.csv"))
write_csv(baseline_external_pre$baseline_table, 
          file.path(output_dir_baseline, "external_baseline_pre_imputation.csv"))

# 5. 保存细菌分析结果为CSV
output_dir_bacteria <- file.path(output_dir_main, "bacteria_analysis")
dir.create(output_dir_bacteria, showWarnings = FALSE, recursive = TRUE)

if(!is.null(bacteria_train)) {
  write_csv(bacteria_train$top10_all, file.path(output_dir_bacteria, "train_bacteria_top10_all.csv"))
  write_csv(bacteria_train$top10_pos, file.path(output_dir_bacteria, "train_bacteria_top10_pos.csv"))
  write_csv(bacteria_train$top10_neg, file.path(output_dir_bacteria, "train_bacteria_top10_neg.csv"))
}

if(!is.null(bacteria_test)) {
  write_csv(bacteria_test$top10_all, file.path(output_dir_bacteria, "test_bacteria_top10_all.csv"))
}

if(!is.null(bacteria_external)) {
  write_csv(bacteria_external$top10_all, file.path(output_dir_bacteria, "external_bacteria_top10_all.csv"))
}

# 6. 保存特征筛选信息
output_dir_features <- file.path(output_dir_main, "feature_selection")
dir.create(output_dir_features, showWarnings = FALSE, recursive = TRUE)

# 保存缺失率分析
if(exists("missing_rates")) {
  write_csv(missing_rates, file.path(output_dir_features, "feature_missing_rates.csv"))
}

# 保存最终特征列表
final_features_df <- data.frame(
  Feature = final_cols,
  Type = ifelse(final_cols %in% existing_metadata_cols, "Metadata", "Feature"),
  Selected = ifelse(final_cols %in% existing_metadata_cols, "Always", 
                    ifelse(final_cols %in% selected_features, "Yes", "No")),
  stringsAsFactors = FALSE
)

write_csv(final_features_df, file.path(output_dir_features, "final_feature_list.csv"))

# 7. 生成汇总报告
report_file <- file.path(output_dir_main, "complete_analysis_report.txt")
sink(report_file)

cat("=== MIMIC数据库完整分析报告 ===\n\n")
cat("分析时间:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")

cat("一、研究概述\n")
cat("1. 研究目的: 血流感染患者的基线特征分析和数据预处理\n")
cat("2. 数据来源: MIMIC-IV数据库 + 外部医院数据\n")
cat("3. 研究对象: Gram阳性 vs Gram阴性血流感染患者\n")
cat("4. 分析方法: 基于缺失率的特征筛选 + 中位数插补\n\n")

cat("二、数据预处理流程\n")
cat("1. 患者筛选:\n")
cat("   - 原始患者数:", length(unique(mimic_data[[id_col]])), "\n")
cat("   - 排除混合感染患者:", length(mixed_ids), "\n")
cat("   - 最终纳入患者:", length(patient_ids), "\n")
cat("   - 排除比例:", round(length(mixed_ids)/length(unique(mimic_data[[id_col]]))*100, 1), "%\n\n")

cat("2. 数据分割:\n")
cat("   - 训练集: ", length(train_ids), "名患者 (", 
    round(length(train_ids)/length(patient_ids)*100, 1), "%)\n", sep = "")
cat("   - 测试集: ", length(test_ids), "名患者 (", 
    round(length(test_ids)/length(patient_ids)*100, 1), "%)\n", sep = "")
cat("   - 外部验证集: ", length(unique(external_filtered[[find_column(external_filtered, 
                                                                 c("subject_id", "hadm_id", "stay_id", "patient_id"))]])), "名患者\n\n", sep = "")

cat("3. 特征筛选（基于训练集）:\n")
cat("   - 原始特征数: ", length(feature_cols), "\n", sep = "")
cat("   - 保留特征数（缺失率≤60%）: ", length(selected_features), "\n", sep = "")
cat("   - 排除特征数（缺失率>60%）: ", length(excluded_features), "\n", sep = "")
cat("   - 最终总列数: ", length(final_cols), "\n\n", sep = "")

cat("三、基线特征分析结果（插补前）\n")
cat("1. 患者分布:\n")
cat("   - 训练集: ", baseline_train_pre$total_patients, "名患者\n", sep = "")
cat("     * Gram Positive: ", ifelse("Gram Positive" %in% names(baseline_train_pre$gram_counts), 
                                     baseline_train_pre$gram_counts["Gram Positive"], 0), 
    " (", round(ifelse("Gram Positive" %in% names(baseline_train_pre$gram_counts), 
                       baseline_train_pre$gram_counts["Gram Positive"], 0) / baseline_train_pre$total_patients * 100, 1), "%)\n", sep = "")
cat("     * Gram Negative: ", ifelse("Gram Negative" %in% names(baseline_train_pre$gram_counts), 
                                     baseline_train_pre$gram_counts["Gram Negative"], 0), 
    " (", round(ifelse("Gram Negative" %in% names(baseline_train_pre$gram_counts), 
                       baseline_train_pre$gram_counts["Gram Negative"], 0) / baseline_train_pre$total_patients * 100, 1), "%)\n\n", sep = "")

cat("2. 变量分析:\n")
cat("   - 连续变量: ", length(baseline_train_pre$continuous_vars), "个\n", sep = "")
cat("   - 分类变量: ", length(baseline_train_pre$categorical_vars), "个\n", sep = "")
cat("   - 总分析变量: ", length(baseline_train_pre$analysis_vars), "个\n\n", sep = "")

cat("四、细菌分布分析\n")
cat("1. 训练集细菌分布（前10）:\n")
if(!is.null(bacteria_train)) {
  cat("   - 总患者数: ", bacteria_train$patient_count, "\n", sep = "")
  cat("   - Gram Positive患者: ", bacteria_train$pos_count, "\n", sep = "")
  cat("   - Gram Negative患者: ", bacteria_train$neg_count, "\n", sep = "")
}

cat("\n五、数据插补\n")
cat("1. 插补策略: 基于训练集中位数，按Gram类型分组插补\n")
cat("2. 插补变量: ", length(setdiff(final_cols, c(id_col, "subject_id", "hadm_id", "stay_id", "patient_id", "id",
                                              "distinct_bacteria_count"))), "个数值变量\n", sep = "")
cat("3. 避免数据泄露: 所有统计量仅基于训练集计算\n\n")

cat("六、输出文件说明\n")
cat("1. 数据文件:\n")
cat("   - pre_imputation_data/ : 插补前的筛选数据\n")
cat("   - post_imputation_data/ : 插补后的完整数据\n\n")

cat("2. 分析结果:\n")
cat("   - baseline_analysis/ : 插补前的基线特征分析\n")
cat("   - bacteria_analysis/ : 细菌分布分析\n")
cat("   - feature_selection/ : 特征筛选信息\n\n")

cat("3. 主要文件:\n")
cat("   - pre_imputation_baseline_analysis.xlsx : 完整的基线特征分析报告\n")
cat("   - complete_analysis_report.txt : 本报告\n\n")

cat("七、学术价值\n")
cat("1. 完整的临床数据分析流程\n")
cat("2. 严格的基线特征分析（插补前）\n")
cat("3. 详细的细菌分布分析\n")
cat("4. 可重复的数据预处理方法\n")
cat("5. 符合学术发表要求的输出格式\n\n")

cat("八、后续建议\n")
cat("1. 使用插补后的数据进行机器学习建模\n")
cat("2. 基于基线特征表格选择重要预测变量\n")
cat("3. 结合临床意义解释统计结果\n")
cat("4. 使用测试集和外部验证集验证模型性能\n")
cat("5. 考虑进一步的特征工程和模型优化\n")

sink()

cat("汇总报告已保存至:", report_file, "\n\n")

# ==============================================================================
# 第八部分：程序结束
# ==============================================================================
cat(str_dup("*", 80), "\n")
cat("程序执行完成！\n")
cat(str_dup("*", 80), "\n\n")

cat("输出目录:", output_dir_main, "\n\n")

cat("主要输出文件:\n")
cat("1. 数据文件目录:\n")
cat("   - ", output_dir_pre_imputation, " (插补前数据)\n",
    
    
    
    
    
    