# File: process_occupation_data.R
# Summary: This file combines the BLS employment data with O*NET Occupation data
#          and includes exploratory code

rm(list = ls())
package_list <- c("fixest", "lubridate","dplyr", "ggplot2", "zoo","DescTools",
                  "tidyverse", "jtools", "gridExtra", "xtable", "Hmisc", "readxl",
                  "tidyr", "ggpubr", "datasets", "openxlsx","pryr","knitr",
                  "kableExtra","data.table")
lapply(package_list, require, character.only = TRUE)
cluster <- 1
options(scipen = 999)

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
## Cross Walks ########
xw <- fread("input/soc_xwalk/2010_to_2019_Crosswalk.csv")
names(xw) <- fix_names(names(xw))

xw_soc <- fread("input/soc_xwalk/2019_to_SOC_Crosswalk.csv")
names(xw_soc) <- fix_names(names(xw_soc))

xw[xw_soc, 
   `:=`(occ_code=i.2018_soc_code,
        occ_title=i.2018_soc_title),
   on=.(onetsoc_2019_code=onetsoc_2019_code)]

xw[,occ_code:=fifelse(occ_code %in% c("11-2032","11-2033"),"11-2031",occ_code)]
xw[,occ_code:=fifelse(occ_code %in% c("11-3012","11-3013"),"11-3011",occ_code)]
xw[,occ_code:=fifelse(occ_code %in% c("13-2022","13-2023"),"13-2021",occ_code)]
xw[,occ_code:=fifelse(occ_code %in% c("19-3033","19-3034"),"19-3031",occ_code)]
xw[,occ_code:=fifelse(occ_code %in% c("19-4012","19-4013"),"19-4011",occ_code)]
xw[,occ_code:=fifelse(occ_code %in% c("25-2055","25-2056"),"25-2052",occ_code)]
xw[,occ_code:=fifelse(occ_code %in% c("29-2043","29-2042"),"29-2041",occ_code)]
xw[,occ_code:=fifelse(occ_code=="51-2091","51-2051",occ_code)]

soc_2010_2018 <- setDT(read_excel(path="input/soc_xwalk/soc_2010_to_2018_crosswalk.xlsx",skip=8))
names(soc_2010_2018) <- fix_names(names(soc_2010_2018))
setnames(soc_2010_2018,
         c("2010_soc_code","2010_soc_title","2018_soc_code","2018_soc_title"),
         c("occ_code","occ_title","occ_code_2018","occ_title_2018"))

## BLS ########
### Unzip files
for (i in 18:23) {
  path <- paste0("input/bls/","oesm",i,"all.zip")
  outdir <- paste0("input/bls/unzipped")
  
  unzip(zipfile=path,exdir=outdir)
}

### Read in excels
j<-1
infiles <- list()
for (i in 18:23) {
  print(i)
  file <- paste0("input/bls/unzipped/oesm",i,"all/all_data_M_20",i,".xlsx")
  sht <- paste0("All May 20",i," data")
  
  dt <- setDT(read_excel(path=file,col_types="text"))
  dt[,year:=paste0("20",i)]
  names(dt) <- fix_names(names(dt))
  
  
  infiles[[j]] <- dt
  j <- j+1
  gc()
}

dt <- rbindlist(infiles,fill=TRUE)

### correct occupation codes so no detailed code ends w/ 0 & no broad code ends w/ 1
dt[,`:=`(maj=substr(occ_code,1,2),
         min=substr(occ_code,4,4),
         brd=substr(occ_code,5,6),
         det=substr(occ_code,7,7))]
dt[,det:=fifelse(det=="0" & o_group=="detailed","1",det)]
dt[,occ_code:=paste0(maj,"-",min,brd,det)]

### adjust occupation codes
dt[,occ_code:=fifelse(occ_code %in% c("11-2032","11-2033"),"11-2031",occ_code)]
dt[,occ_code:=fifelse(occ_code %in% c("11-3012","11-3013"),"11-3011",occ_code)]
dt[,occ_code:=fifelse(occ_code %in% c("13-2022","13-2023"),"13-2021",occ_code)]
dt[,occ_code:=fifelse(occ_code %in% c("19-3033","19-3034"),"19-3031",occ_code)]
dt[,occ_code:=fifelse(occ_code %in% c("19-4012","19-4013"),"19-4011",occ_code)]
dt[,occ_code:=fifelse(occ_code %in% c("25-2055","25-2056"),"25-2052",occ_code)]
dt[,occ_code:=fifelse(occ_code %in% c("29-2043","29-2042"),"29-2041",occ_code)]
dt[,occ_code:=fifelse(occ_code=="51-2091","51-2051",occ_code)]

### loop over 4-digit through 6-digit NAICS granularity
for (i in 4:6) {
  ### keep most inclusive ownership level, detailed occupation group, US-wide employment
  i_group_str <- paste0(i,"-digit")
  bls <- copy(dt)
  bls <- bls[area=="99"]
  bls <- bls[o_group=="detailed"]
  bls <- bls[i_group==i_group_str]
  bls[,max_own_code:=max(as.numeric(own_code)),by=.(naics,occ_code,occ_title,year)]
  bls <- bls[own_code==as.character(max_own_code)]
  
  ### standardize occupation titles
  occ_titles <- bls[,.N,by=.(occ_code,occ_title,year)]
  occ_codes <- c("11-2031","11-3011","13-2021","19-3031","19-4011","25-2052","29-2041")
  occ_titles[order(occ_code,year),
             occ_title:=fifelse(occ_code %in% occ_codes,first(occ_title),occ_title),
             by=(occ_code)]
  occ_titles[order(occ_code,year),
             occ_title:=fifelse(occ_code=="51-2091",last(occ_title),occ_title),
             by=(occ_code)]
  
  bls[occ_titles,occ_title:=i.occ_title,on=.(occ_code=occ_code)]
  
  ### sum across standardized occupations
  bls[,`:=`(tot_emp=as.numeric(tot_emp),
            a_mean=as.numeric(a_mean))]
  
  bls <- bls[,.(tot_emp=sum(tot_emp,na.rm=TRUE),
                tot_wages=sum(tot_emp*a_mean,na.rm=TRUE),
                avg_salary=mean(a_mean,na.rm=TRUE)),
             by=.(occ_code,occ_title,naics,naics_title,year)]
  
  setcolorder(bls,c("year","naics","naics_title","occ_code","occ_title","tot_emp","tot_wages","avg_salary"))
  bls <- bls[,.(year,naics,naics_title,occ_code,occ_title,tot_emp,tot_wages,avg_salary)]
  
  ### prepend arbitrary character to occ_code (excel reads these as dates otherwise)
  bls[,occ_code:=paste0("X",occ_code)]
  write_csv(bls,file=paste0("output/data/bls_oews_yearly_",i,"_digit.csv"))
  
}

## O*NET ########
dates <- list()
dates["db_22_2_excel"] <- "2018q1"
dates["db_22_3_excel"] <- "2018q2"
dates["db_23_0_excel"] <- "2018q3"
dates["db_23_1_excel"] <- "2018q4"
dates["db_23_2_excel"] <- "2019q1"
dates["db_23_3_excel"] <- "2019q2"
dates["db_24_0_excel"] <- "2019q3"
dates["db_24_1_excel"] <- "2019q4"
dates["db_24_2_excel"] <- "2020q1"
dates["db_24_3_excel"] <- "2020q2"
dates["db_25_0_excel"] <- "2020q3"
dates["db_25_1_excel"] <- "2020q4"
dates["db_25_2_excel"] <- "2021q1"
dates["db_25_3_excel"] <- "2021q2"
dates["db_26_0_excel"] <- "2021q3"
dates["db_26_1_excel"] <- "2021q4"
dates["db_26_2_excel"] <- "2022q1"
dates["db_26_3_excel"] <- "2022q2"
dates["db_27_0_excel"] <- "2022q3"
dates["db_27_1_excel"] <- "2022q4"
dates["db_27_2_excel"] <- "2023q1"
dates["db_27_3_excel"] <- "2023q2"
dates["db_28_0_excel"] <- "2023q3"
dates["db_28_1_excel"] <- "2023q4"

### load in quarterly occupation data
infiles <- list()
for (file in names(dates)) {
  
  print(file)
  path <- paste0("input/onet/",file,".zip")
  outdir <- "input/onet/unzipped"
  unzip(zipfile=path,exdir=outdir)
  date <- dates[[file]]
  
  ### universe of occupations
  occupations <- setDT(read_excel(path=paste0("input/onet/unzipped/",file,"/Occupation Data.xlsx"),col_types="text"))
  names(occupations) <- fix_names(names(occupations))
  occupations <- occupations[,.(onetsoc_code,title)]
  occupations[,`:=`(onetsoc_code=str_squish(onetsoc_code))]
  
  ### task statements
  tasks <- setDT(read_excel(path=paste0("input/onet/unzipped/",file,"/Task Statements.xlsx"),col_types="text"))
  names(tasks) <- fix_names(names(tasks))
  tasks <- tasks[,.(onetsoc_code,task_id,task,task_type)]
  tasks[,`:=`(onetsoc_code=str_squish(onetsoc_code),
              task_id=str_squish(task_id))]
  
  ### task-DWA mapping
  dwas <- setDT(read_excel(path=paste0("input/onet/unzipped/",file,"/Tasks to DWAs.xlsx"),col_types="text"))
  names(dwas) <- fix_names(names(dwas))
  dwas <- dwas[,.(onetsoc_code,task_id,dwa_id,dwa_title)]
  dwas[,`:=`(onetsoc_code=str_squish(onetsoc_code),
             dwa_id=str_squish(dwa_id),
             task_id=str_squish(task_id))]
  
  ### task weights
  task_ratings <- setDT(read_excel(path=paste0("input/onet/unzipped/",file,"/Task Ratings.xlsx"),col_types="text"))
  names(task_ratings) <- fix_names(names(task_ratings))
  
  task_ratings <- task_ratings[,.(onetsoc_code,task_id,scale_id,category,data_value)]
  task_ratings[, scale_id:=fifelse(scale_id=="FT",paste(scale_id,category,sep="_"),scale_id)]
  task_ratings[,category:=NULL]
  task_ratings <- dcast(task_ratings,onetsoc_code+task_id~scale_id,value.var="data_value")
  setnames(task_ratings,
           c("FT_1","FT_2","FT_3","FT_4","FT_5","FT_6","FT_7","IM","RT"),
           c("yearly_or_less","more_than_yearly","more_than_monthly",
             "more_than_weekly","daily","several_times_daily",
             "hourly_or_more","task_importance","task_relevance"))
  task_ratings[,`:=`(onetsoc_code=str_squish(onetsoc_code),
                     task_id=str_squish(task_id))]
  
  ### DWA to WA mapping
  wa_dwa <- setDT(read_excel(path=paste0("input/onet/unzipped/",file,"/DWA Reference.xlsx"),col_types="text"))
  names(wa_dwa) <- fix_names(names(wa_dwa))
  wa_dwa <- wa_dwa[,.(element_id,element_name,dwa_id)]
  setnames(wa_dwa,c("element_id","element_name"),c("wa_id","wa_name"))
  wa_dwa[,`:=`(wa_id=str_squish(wa_id),
               dwa_id=str_squish(dwa_id))]
  
  ### WA weights
  wa_ratings <- setDT(read_excel(path=paste0("input/onet/unzipped/",file,"/Work Activities.xlsx"),col_types="text"))
  names(wa_ratings) <- fix_names(names(wa_ratings))
  
  wa_ratings <- wa_ratings[scale_id=="IM"]
  wa_ratings <- wa_ratings[,.(onetsoc_code,element_id,data_value)]
  setnames(wa_ratings,c("element_id","data_value"),c("wa_id","wa_importance"))
  wa_ratings[,`:=`(onetsoc_code=str_squish(onetsoc_code),
                   wa_id=str_squish(wa_id))]
  
  # combine -> using merge.data.table() b/c some merges result in expansions
  dt <- merge.data.table(occupations,tasks,by="onetsoc_code")
  dt <- merge.data.table(dt,task_ratings,by=c("onetsoc_code","task_id"),all.x=TRUE)
  dt <- merge.data.table(dt,dwas,by=c("onetsoc_code","task_id"),all.x=TRUE)
  dt <- merge.data.table(dt,wa_dwa,by=c("dwa_id"),all.x=TRUE)
  dt <- merge.data.table(dt,wa_ratings,by=c("onetsoc_code","wa_id"),all.x=TRUE)
  
  # clean strings
  dt[,`:=`(title=str_squish(title),
           task=str_squish(task),
           task_type=str_squish(task_type),
           dwa_title=str_squish(dwa_title),
           wa_name=str_squish(wa_name))]
  
  dt[,rel:=date]
  dt[,year:=substr(date,1,4)]
  infiles[[file]] <- dt
  
  gc()
  
}
onet <- rbindlist(infiles)

### standarize O*Net occupation codes
# 2010 -> 2019 O*Net occupation mapping
onet[xw,
     `:=`(onetsoc_2019_code=i.onetsoc_2019_code,
          onetsoc_2019_title=i.onetsoc_2019_title),
     on=.(onetsoc_code=onetsoc_2010_code)]
onet[,`:=`(onetsoc_code=fifelse(year==2018,onetsoc_2019_code,onetsoc_code),
           onetsoc_title=fifelse(year==2018,onetsoc_2019_title,title))]
onet[,`:=`(onetsoc_2019_code=NULL,onetsoc_2019_title=NULL)]

# 2019 O*Net occupation code -> 2018 SOC code
tmp_xw <- xw[,.N,by=.(onetsoc_2019_code,occ_code,occ_title)]
onet[tmp_xw,
     `:=`(occ_code=i.occ_code,
          occ_title=i.occ_title),
     on=.(onetsoc_code=onetsoc_2019_code)]

onet[is.na(occ_code),.N,by=year] ## only missing codes are 15-1299.04-0.7 -> manually map to 15-1299
onet[onetsoc_code %in% c("15-1299.04","15-1299.05","15-1299.06","15-1299.07"),
     `:=`(occ_code="15-1299",occ_title="Computer Occupations, All Other")]

### Export yearly & quarterly excerpts
onet[,`:=`(onetsoc_title=NULL,onetsoc_code=NULL,title=NULL)]
setcolorder(onet,c("year","rel","occ_code","occ_title","task_id","task","task_type",
                   "wa_id","wa_name","dwa_id","dwa_title"))

cols <- c("yearly_or_less","more_than_yearly","more_than_monthly","more_than_weekly",
          "daily","several_times_daily","hourly_or_more","task_importance","task_relevance",
          "wa_importance")
onet[,c(cols):=lapply(.SD,function(x) as.numeric(x)),.SDcols=cols]

onet_yearly <- onet[,.(yearly_or_less=mean(yearly_or_less,na.rm=TRUE),
                       more_than_yearly=mean(more_than_yearly,na.rm=TRUE),
                       more_than_monthly=mean(more_than_monthly,na.rm=TRUE),
                       more_than_weekly=mean(more_than_weekly,na.rm=TRUE),
                       daily=mean(daily,na.rm=TRUE),
                       several_times_daily=mean(several_times_daily,na.rm=TRUE),
                       hourly_or_more=mean(hourly_or_more,na.rm=TRUE),
                       task_importance=mean(task_importance,na.rm=TRUE),
                       task_relevance=mean(task_relevance,na.rm=TRUE),
                       wa_importance=mean(wa_importance,na.rm=TRUE)),
                    by=.(year,occ_code,occ_title,task_id,task,task_type,
                         wa_id,wa_name,dwa_id,dwa_title)]

write_csv(onet,file="output/data/onet_occupations_quarterly.csv")
write_csv(onet_yearly,file="output/data/onet_occupations_yearly.csv")

# CLEAN UP ####
## BLS Clean Up ########
bls_files <- list.files("input/bls/unzipped",full.names=TRUE)
for (file in bls_files) {
  unlink(file,recursive=TRUE)
}

## O*Net clean up ########
onet_files <- list.files("input/onet/unzipped",full.names=TRUE)
for (file in onet_files) {
  unlink(file,recursive=TRUE)
}



