# File: radiology_task_analysis.R
# Summary: This file approximates the amount of time required to complete a task
#          in the O*Net occupation data

rm(list = ls())
package_list <- c("fixest", "lubridate","dplyr", "ggplot2", "zoo","DescTools",
                  "tidyverse", "jtools", "gridExtra", "xtable", "Hmisc", "readxl",
                  "tidyr", "ggpubr", "datasets", "openxlsx","pryr","knitr",
                  "kableExtra","httr","data.table")
lapply(package_list, require, character.only = TRUE)
cluster <- 1
options(scipen = 999)
api_key <- ""

if (Sys.info()["sysname"]=="Windows") {
  setwd("C:/Users/spomeran/Dropbox (MIT)/Sam-Mert Share2/ai-exposure")  
} else if (Sys.info()["sysname"]=="Linux") {
  setwd("/home/spomeran/ai-exposure")
}

fix_names <- function(name){
  name  <- gsub(" ", "_", name)
  remote <- c(" ", "\\$", "/", "%", "-", "\\(", "\\)","\\*")
  for(i in remote){
    name  <- gsub(i, "", name)
  }
  name  <- gsub("__", "_", name)
  name  <- tolower(name)
  return(name)
}

# LOAD DATA ####
onet <- fread("output/data/onet_occupations_yearly.csv")

# RESTRICTIONS ####
# keep radiologists
onet <- onet[occ_code=="29-1224"]

# standardize task IDs across time
onet[order(task,year),task_id:=last(task_id),by=task]
onet[task_id==17152,task_id:=22736] # these tasks are the same but differ by only a comma

# keep tasks that appear in 2023
tasks_2023 <- onet[year==2023,.N,by=task_id][,task_id]
onet <- onet[task_id %in% tasks_2023]

# drop if missing information
onet <- onet[!is.na(yearly_or_less)]

# aggregate to task level, carry forward non-missing values
onet <- onet[order(year),.(yearly_or_less=last(yearly_or_less),
                           more_than_yearly=last(more_than_yearly),
                           more_than_monthly=last(more_than_monthly),
                           more_than_weekly=last(more_than_weekly),
                           daily=last(daily),
                           several_times_daily=last(several_times_daily),
                           hourly_or_more=last(hourly_or_more),
                           task_importance=last(task_importance),
                           task_relevance=last(task_relevance)),
             by=.(occ_code,occ_title,task_id,task,task_type)]

# CALCULATE AGGREGATE TIMES ####
# fork out importance/relevance
weights <- onet[,.N,by=.(task_id,task_importance,task_relevance)]

# keep tasks w/ frequencies
freqs <- onet[!is.na(yearly_or_less)]
freqs <- freqs[,.(task,task_id,task_type,yearly_or_less,more_than_yearly,
                  more_than_monthly,more_than_weekly,daily,several_times_daily,
                  hourly_or_more)]

# reshape
freqs_long <- melt(freqs,id.vars=c("task","task_id","task_type"),
                   measure.vars=c("yearly_or_less","more_than_yearly","more_than_monthly",
                                  "more_than_weekly","daily","several_times_daily","hourly_or_more"),
                   variable.name="freq_bin",value.name="share")

# calculate avg freq bin -> assuming 261 working days in a year
freqs_long[,share:=share/100]
freqs_long[freq_bin=="yearly_or_less",`:=`(freq_val=1,freq_bin_val=1)] # once per year
freqs_long[freq_bin=="more_than_yearly",`:=`(freq_val=3,freq_bin_val=2)] # assuming this is 3 times yearly
freqs_long[freq_bin=="more_than_monthly",`:=`(freq_val=18,freq_bin_val=3)] # 1-2 times per month
freqs_long[freq_bin=="more_than_weekly",`:=`(freq_val=78,freq_bin_val=4)] # 1-2 times per week
freqs_long[freq_bin=="daily",`:=`(freq_val=261,freq_bin_val=5)] 
freqs_long[freq_bin=="several_times_daily",`:=`(freq_val=1044,freq_bin_val=6)] # assuming this is 4-times daily
freqs_long[freq_bin=="hourly_or_more",`:=`(freq_val=6264,freq_bin_val=7)] # once per hourly

freqs_avg <- freqs_long[,.(avg_freq_val=sum(freq_val*share),
                           avg_freq_bin=sum(freq_bin_val*share)),
                        by=.(task_id,task_name=task,task_type)]
freqs_avg[,rel_freq:=avg_freq_val/sum(avg_freq_val)]
freqs_avg[,hours_per_yr:=rel_freq*2080]

freqs_avg[avg_freq_bin<=1.2,freq_bin:="Yearly or Less"]
freqs_avg[avg_freq_bin<1.8 & avg_freq_bin>1.2,freq_bin:="Approximately Yearly"]
freqs_avg[avg_freq_bin<=2.2 & avg_freq_bin>1.8,freq_bin:="more than yearly"]
freqs_avg[avg_freq_bin<2.8 & avg_freq_bin>2.2,freq_bin:="More than Yearly, less than Monthly"]
freqs_avg[avg_freq_bin<=3.2 & avg_freq_bin>=2.8,freq_bin:="more than monthly"]
freqs_avg[avg_freq_bin<3.8 & avg_freq_bin>3.2,freq_bin:="More than monthly, less than weekly"]
freqs_avg[avg_freq_bin<=4.2 & avg_freq_bin>=3.8,freq_bin:="more than weekly"]
freqs_avg[avg_freq_bin<4.8 & avg_freq_bin>4.2,freq_bin:="More than weekly, less than daily"]
freqs_avg[avg_freq_bin<=5.2 & avg_freq_bin>=4.8,freq_bin:="daily"]
freqs_avg[avg_freq_bin<5.8 & avg_freq_bin>5.2,freq_bin:="a few times daily"]
freqs_avg[avg_freq_bin<=6.2 & avg_freq_bin>=5.8,freq_bin:="several times daily"]
freqs_avg[avg_freq_bin<6.8 & avg_freq_bin>6.2,freq_bin:="less than hourly"]
freqs_avg[avg_freq_bin>=6.8,freq_bin:="hourly or more"]

task_summary <- copy(freqs_avg)
setnames(task_summary, "avg_freq_bin","freq_bin_val")
task_summary <- task_summary[,.(task_id,task_name,task_type,freq_bin,freq_bin_val,hours_per_yr)]

# ChatGPT API ####
# get list of tasks
tasks <- task_summary[,task_name]

pre1 <- "Consider this occupation: Radiologist \n"
pre2 <- "Part of this occupation is this task: "

time_to_complete <- "How many minutes would it take to do this task, on average? 
Think step-by-step. Return a precise estimated time (i.e., not a range of times) and formatted like this: {answer: <put estimated time here>}{comment: <put explanation here>}"

time_w_ai <- "Imagine this task can be automated with the use of artificial intelligence. The worker is now responsible for inspecting the output of the task.

What is the ratio of time to evaluate this task relative to the time it takes to do it?
E.g., if it takes just as long to inspect the output as to do the task, then 1.
If the task output could be inspected in 5 minutes but the task takes 10 minutes, then 0.5.

Return a precise estimate ranging from 0 to 1 and formatted like this: {answer: <put estimated time here>}{comment: <put explanation here>}"

prob_auto_prmpt <- "Imagine a radiologist is implementing artificial intelligence tools to simplify this task.

What is the likelihood that this task can be fully automated using AI?
E.g., if the task can be fully automated with absolute certainty, then 1. If the task has a 50% chance of being fully automated, then 0.5.

Return an estimated likelihood as a decimal ranging from 0 to 1 and formatted like this: {answer: <put estimated time here>}{comment: <put explanation here>}"
  
share_auto_prmpt <- "Imagine a radiologist is implementing artificial intelligence tools to simplify this task.

What amount of work involved in this task could be automated with AI? 
E.g., if the portion of the work for this task that could be automated accounts for 50% of the total time required, then 0.5. If the task could be fully automated, then 1.

Return an estimated likelihood as a decimal ranging from 0 to 1 and formatted like this: {answer: <put estimated time here>}{comment: <put explanation here>}"

ttc_sys_msg <- "You are answering questions as if you were a human. Do not break character. You are an agent with the following persona: a medical doctor approximating the amount of time required to complete tasks associated with the practice of medicine."
ai_xprt_sys_msg <- "You are answering questions as if you were a human. Do not break character. You are an agent with the following persona: an expert in medical practice and artificial intelligence, evaluating the potential assistance artificial intelligence can provide medical doctors in their daily tasks."

task_summary[,`:=`(automation_share_raw="",
                   automation_prob_raw="",
                   comp_time_mins_raw="",
                   review_shr_raw="")]

temp <- 1
temp_str <- temp*100
for (task in tasks) {
  print(task)
  
  ### Completion Times
  # time to complete task
  prompt <- paste0(pre1,pre2,task," \n",time_to_complete)

  res <- POST(
    url = "https://api.openai.com/v1/chat/completions",
    add_headers(Authorization = paste("Bearer", api_key)),
    content_type_json(),
    encode = "json",
    body = list(
      model = "gpt-4-turbo",
      temperature = temp,
      seed=265,
      messages = list(list(role = "system",
                           content = ttc_sys_msg),
                      list(role = "user",
                           content = prompt))
    )
  )
  ans <- content(res)$choices[[1]]$message$content
  task_summary[,comp_time_mins_raw:=fifelse(task_name==task,ans,comp_time_mins_raw)]
  
  # time to inspect ai output
  prompt <- paste0(pre1,pre2,task," \n",time_w_ai)

  res <- POST(
    url = "https://api.openai.com/v1/chat/completions",
    add_headers(Authorization = paste("Bearer", api_key)),
    content_type_json(),
    encode = "json",
    body = list(
      model = "gpt-4-turbo",
      temperature = temp,
      seed=265,
      messages = list(list(role = "system",
                           content = ai_xprt_sys_msg),
                      list(role = "user",
                           content = prompt))
    )
  )
  ans <- content(res)$choices[[1]]$message$content
  task_summary[,review_shr_raw:=fifelse(task_name==task,ans,review_shr_raw)]
  
  ### AI contributions
  # likelihood that task can be fully automated
  prompt <- paste0(pre1,pre2,task," \n",prob_auto_prmpt)

  res <- POST(
    url = "https://api.openai.com/v1/chat/completions",
    add_headers(Authorization = paste("Bearer", api_key)),
    content_type_json(),
    encode = "json",
    body = list(
      model = "gpt-4-turbo",
      temperature = temp,
      seed=265,
      messages = list(list(role = "system",
                           content = ai_xprt_sys_msg),
                      list(role = "user",
                           content = prompt))
    )
  )
  ans <- content(res)$choices[[1]]$message$content
  task_summary[,automation_prob_raw:=fifelse(task_name==task,ans,automation_prob_raw)]
  
  # share of task that can be automated
  prompt <- paste0(pre1,pre2,task," \n",share_auto_prmpt)

  res <- POST(
    url = "https://api.openai.com/v1/chat/completions",
    add_headers(Authorization = paste("Bearer", api_key)),
    content_type_json(),
    encode = "json",
    body = list(
      model = "gpt-4-turbo",
      temperature = temp,
      seed=265,
      messages = list(list(role = "system",
                           content = ai_xprt_sys_msg),
                      list(role = "user",
                           content = prompt))
    )
  )
  ans <- content(res)$choices[[1]]$message$content
  task_summary[,automation_share_raw:=fifelse(task_name==task,ans,automation_share_raw)]
  
}

# clean up responses
task_summary[,`:=`(automation_share=as.numeric(gsub(".*\\{answer:\\s*([0-9]+\\.?[0-9]*).*", "\\1", automation_share_raw)),
                   automation_prob=as.numeric(gsub(".*\\{answer:\\s*([0-9]+\\.?[0-9]*).*", "\\1", automation_prob_raw)),
                   comp_time_mins=as.numeric(gsub(".*\\{answer:\\s*([0-9]+\\.?[0-9]*).*", "\\1", comp_time_mins_raw)),
                   review_shr=as.numeric(gsub(".*\\{answer:\\s*([0-9]+\\.?[0-9]*).*", "\\1", review_shr_raw)))]

# merge in weights
task_summary[weights,`:=`(task_importance=i.task_importance,
                          task_relevance=i.task_relevance),
             on=.(task_id=task_id)]

write_csv(task_summary[,.(task_id,task_name,task_type,freq_bin,freq_bin_val,hours_per_yr,
                          comp_time_mins,review_shr,automation_share,automation_prob,
                          task_importance,task_relevance)],
          file=paste0("output/radiology_analysis/radiology_task_summary_temp",temp_str,".csv"))




