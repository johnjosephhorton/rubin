# File: explore_occupation_data.R
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

soc_2010 <- soc_2010_2018[,.N,by=occ_code][,occ_code]
soc_2018 <- soc_2010_2018[,.N,by=occ_code_2018][,occ_code_2018]

soc_2010_only <- soc_2010[!(soc_2010 %in% soc_2018)]
soc_2018_only <- soc_2018[!(soc_2018 %in% soc_2010)]

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
# 2015-2019 missing prim_state, pct_rpt

dt <- rbindlist(infiles,fill=TRUE)
dt <- dt[area=="99"] # keep country-wide obs

# # standardize 2017 SOC codes to 2018 version
# dt[soc_2010_2018,
#    `:=`(occ_code_2018=i.occ_code_2018,
#                       occ_title_2018=i.occ_title_2018),
#    on=.(occ_code=occ_code,occ_title=occ_title)]
# 
# dt[year==2017,.N]
# dt[year==2017 & is.na(occ_code_2018),.N,by=o_group]

### NOTES:
# keeping years 2018 through 2023 b/c 2015 & 2016 are missing industry groups (4-digit vs. 5-digit, etc.) & 
# new SOC classification scheme begins in 2018
# keeping industry information up to 4-digits -> 5-/6-digit codes do not include all occupations
# Using tot_emp to measure employment
# keeping area_title==U.S. for nationwide employment figures
# keeping detailed SOC codes to avoid double counting

# TO DO 
# ... should also figure out why some 6-digit industry codes have fewer occupations, but that can wait

### look at relationship between o-groups
# keep cross-industry for now -> all employment
x_ind <- dt[i_group=="cross-industry"]

x_ind[,.N,by=o_group]

# split SOC code into constituent element
x_ind[str_length(occ_code)!=7,.N]==0
x_ind <- x_ind[o_group!="total"]
x_ind[,`:=`(maj=substr(occ_code,1,2),
            min=substr(occ_code,4,4),
            brd=substr(occ_code,5,6),
            det=substr(occ_code,7,7))]

# number of unique subcodes within larger grouping
x_ind[,min_ct:=uniqueN(min),by=maj]
x_ind[,brd_ct:=uniqueN(brd),by=.(maj,min)]
x_ind[,det_ct:=uniqueN(det),by=.(maj,min,brd)]

x_ind[,.N,by=min_ct][order(min_ct)]
x_ind[o_group!="major",.N,by=brd_ct][order(brd_ct)]
x_ind[o_group!="major" & o_group!="minor",.N,by=det_ct][order(det_ct)]
# view(x_ind[det_ct==1 & !(o_group %in% c("major","minor"))])
#NOTE: In some cases, o_group=="broad" does not contain minor categories b/c it is the most granular
tmp <- x_ind[det_ct==1 & o_group=="broad",.N,by=occ_code][,occ_code]
length(tmp) == x_ind[occ_code %in% tmp & o_group=="detailed",.N,by=occ_code][,.N]
# all of these codes appear twice as broad and detailed -> no detailed occ_code should end with 0

### correct occupation codes so no detailed code ends w/ 0 & no broad code ends w/ 1
dt[,`:=`(maj=substr(occ_code,1,2),
         min=substr(occ_code,4,4),
         brd=substr(occ_code,5,6),
         det=substr(occ_code,7,7))]
dt[,det:=fifelse(det=="0" & o_group=="detailed","1",det)]
dt[,occ_code:=paste0(maj,"-",min,brd,det)]

### number of occupations by industry granularity
dt[o_group=="detailed",.N,by=occ_code][,.N]
dt[o_group=="detailed",.N,by=.(occ_code,year)][,.N,by=year] # not every occupation shows up in every year
# view(dt[o_group=="detailed",.N,by=.(occ_code,year)][,.N,by=occ_code])

tmp<-dt[o_group=="detailed" & i_group=="4-digit",.N,by=.(occ_code,occ_title)]
tmp[,code_ct:=uniqueN(occ_code),by=occ_title]
tmp[,title_ct:=uniqueN(occ_title),by=occ_code]
tmp[,summary(code_ct)]
tmp[,summary(title_ct)]

# view(dt[occ_code %in% soc_2010_only & o_group=="detailed" & year>2020,.N,by=occ_code])

dt[occ_code %in% soc_2018_only & o_group=="detailed",.N,by=year]
dt[occ_code %in% soc_2010 & o_group=="detailed",.N,by=year]
dt[occ_code %in% soc_2018 & o_group=="detailed",.N,by=year]

# view(dt[occ_code %in% soc_2010_only & o_group=="detailed" & year>2018,.N,by=occ_code])
# view(dt[i_group=="cross-industry" & occ_code=="13-2021"])
# view(dt[i_group=="cross-industry" & occ_code=="13-2022"])
# view(dt[i_group=="cross-industry" & occ_code=="13-2023"])
# 
# view(dt[i_group=="cross-industry" & occ_code=="25-2052"])
# view(dt[i_group=="cross-industry" & occ_code=="25-2055"])
# view(dt[i_group=="cross-industry" & occ_code=="25-2056"])
# 
# view(dt[i_group=="cross-industry" & occ_code=="51-2091"])
# view(dt[i_group=="cross-industry" & occ_code=="51-2051"])

# view(dt[occ_code %in% soc_2010_only & o_group=="detailed" & year>2018,.N,by=occ_code])
tmp <- dt[occ_code %in% soc_2010_only & o_group=="detailed" & year>2018,.N,by=occ_code][,occ_code]

# view(dt[occ_code %in% soc_2018_only & o_group=="detailed" & year<2018,.N,by=occ_code])

# NEED TO MANUALLY ADJUST THE FOLLOWING 2010 OCC_CODES THAT SHOULDN'T SHOW UP:
# 11-2031 <- 11-2032 & 11-2033
# 11-3011 <- 11-3012 & 11-3013
# 13-2021 <- 13-2022 & 13-2023
# 19-3031 <- 19-3033 & 19-3034
# 19-4011 <- 19-4012 & 19-4013
# 25-2052 <- 25-2055 & 25-2056
# 29-2041 <- 29-2042 & 29-2043
# 51-2091 -> 51-2051

# these adjustments revert to the most inclusive codes when not standardized across time
dt[,occ_code:=fifelse(occ_code %in% c("11-2032","11-2033"),"11-2031",occ_code)]
dt[,occ_code:=fifelse(occ_code %in% c("11-3012","11-3013"),"11-3011",occ_code)]
dt[,occ_code:=fifelse(occ_code %in% c("13-2022","13-2023"),"13-2021",occ_code)]
dt[,occ_code:=fifelse(occ_code %in% c("19-3033","19-3034"),"19-3031",occ_code)]
dt[,occ_code:=fifelse(occ_code %in% c("19-4012","19-4013"),"19-4011",occ_code)]
dt[,occ_code:=fifelse(occ_code %in% c("25-2055","25-2056"),"25-2052",occ_code)]
dt[,occ_code:=fifelse(occ_code %in% c("29-2043","29-2042"),"29-2041",occ_code)]
dt[,occ_code:=fifelse(occ_code=="51-2091","51-2051",occ_code)]

dt[o_group=="detailed",.(occ_ct=uniqueN(occ_code)),by=i_group] # 5 and 6 digit i_groups seem incomplete

### number of occupations by industry granularity (atfer standardizing)
# drop 2018 codes that split up 2010 codes (i.e. 2010:2018 is 1:m)
soc_2010_2018[,ct_2018:=uniqueN(occ_code_2018),by=occ_code]
tmp_xw <- soc_2010_2018[ct_2018==1]

# drop time invariant codes -> no need to update
tmp_xw <- tmp_xw[occ_code!=occ_code_2018]

# update occ_codes
dt[tmp_xw,`:=`(occ_code_2018=i.occ_code_2018),on=.(occ_code=occ_code)]
dt[,occ_code:=fifelse(!is.na(occ_code_2018),occ_code_2018,occ_code)]
dt[o_group=="detailed",.N,by=occ_code][,.N]
dt[o_group=="detailed",.N,by=.(occ_code,year)][,.N,by=year] # not every occupation shows up in every year
# view(dt[o_group=="detailed",.N,by=.(occ_code,year)][,.N,by=occ_code])

##issue: multiple codes are split into multiple new codes (i.e., 15-1131/33 are both split into 15-1253)
# so can't update them back also can't backfill them -> these will be dropped for now b/c can't merge with
# O*Net data

dt[,max_own_code:=max(as.numeric(own_code)),by=.(naics,occ_code,occ_title,year)]
bls <- dt[area=="99" & o_group=="detailed" & i_group=="4-digit" & own_code==as.character(max_own_code)]

occ_titles <- bls[,.N,by=.(occ_code,occ_title,year)]
occ_codes <- c("11-2031","11-3011","13-2021","19-3031","19-4011","25-2052","29-2041")
occ_titles[order(occ_code,year),
           occ_title:=fifelse(occ_code %in% occ_codes,first(occ_title),occ_title),
           by=(occ_code)]
occ_titles[order(occ_code,year),
           occ_title:=fifelse(occ_code=="51-2091",last(occ_title),occ_title),
           by=(occ_code)]
bls[occ_titles,occ_title:=i.occ_title,on=.(occ_code=occ_code)]

bls[,ct:=.N,by=.(naics,occ_code,year)][ct==2,.N]
bls[,ct:=.N,by=.(naics,occ_code,year)][ct==2 & is.na(occ_code_2018),.N,by=occ_code]
# view(bls[,ct:=.N,by=.(naics,occ_code,year)][ct==2 & is.na(occ_code_2018)][order(naics,occ_code)])


# sum across standardized occupations
bls[,`:=`(tot_emp2=as.numeric(tot_emp),
          a_mean2=as.numeric(a_mean))]

bls <- bls[,.(tot_emp=sum(tot_emp2,na.rm=TRUE),
              tot_wages=sum(tot_emp2*a_mean2,na.rm=TRUE)),
           by=.(occ_code,occ_title,naics,naics_title,year)]

### clean up

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

# for now, ignore weighting of tasks b/c not concerned with industry exposure at this initial stage of cleaning

infiles <- list()
for (file in names(dates)) {
  print(file)
  path <- paste0("input/onet/",file,".zip")
  outdir <- "input/onet/unzipped"
  unzip(zipfile=path,exdir=outdir)
  date <- dates[[file]]
  
  occupations <- setDT(read_excel(path=paste0("input/onet/unzipped/",file,"/Occupation Data.xlsx"),col_types="text"))
  names(occupations) <- fix_names(names(occupations))
  occupations <- occupations[,.(onetsoc_code,title)]
  
  tasks <- setDT(read_excel(path=paste0("input/onet/unzipped/",file,"/Task Statements.xlsx"),col_types="text"))
  names(tasks) <- fix_names(names(tasks))
  tasks <- tasks[,.(onetsoc_code,task_id,task,task_type)]
  
  dwas <- setDT(read_excel(path=paste0("input/onet/unzipped/",file,"/Tasks to DWAs.xlsx"),col_types="text"))
  names(dwas) <- fix_names(names(dwas))
  dwas <- dwas[,.(onetsoc_code,task_id,dwa_id,dwa_title)]
  
  int_merge <- merge.data.table(occupations,tasks,by="onetsoc_code")
  dt <- merge.data.table(int_merge,dwas,by=c("onetsoc_code","task_id"),all.x=TRUE)
  
  dt[,rel:=date]
  dt[,year:=substr(date,1,4)]
  infiles[[file]] <- dt
  
}
onet <- rbindlist(infiles)
onet[,.N,by=.(onetsoc_code,rel,task_id,dwa_id)][N>1,.N]==0
onet[xw,
     `:=`(onetsoc_2019_code=i.onetsoc_2019_code,
          onetsoc_2019_title=i.onetsoc_2019_title),
     on=.(onetsoc_code=onetsoc_2010_code)]

onet[is.na(onetsoc_2019_code),.N,by=year]
onet[is.na(onetsoc_2019_title),.N,by=year]

onet[,`:=`(onetsoc_code=fifelse(year==2018,onetsoc_2019_code,onetsoc_code),
           onetsoc_title=fifelse(year==2018,onetsoc_2019_title,title))]
onet[,`:=`(onetsoc_2019_code=NULL,onetsoc_2019_title=NULL)]

tmp_xw <- xw[,.N,by=.(onetsoc_2019_code,occ_code,occ_title)]

onet[tmp_xw,
     `:=`(occ_code=i.occ_code,
          occ_title=i.occ_title),
     on=.(onetsoc_code=onetsoc_2019_code)]

onet[is.na(occ_code),.N,by=year] ## only missing codes are 15-1299.04-0.7 -> manually map to 15-1299
onet[onetsoc_code %in% c("15-1299.04","15-1299.05","15-1299.06","15-1299.07"),
    `:=`(occ_code="15-1299",occ_title="Computer Occupations, All Other")]

onet <- onet[,.N,by=.(occ_code,occ_title,task_id,task,dwa_id,dwa_title,year,rel)]

## Compare occupation codes in each dataset
onet_occs <- onet[,.N,by=occ_code][,.(occ_code)]
bls_occs <- bls[,.N,by=occ_code][,.(occ_code)]

onet_occs[,.N] # 799 occupation codes
bls_occs[,.N] # 868 occupation codes

int <- fintersect(onet_occs,bls_occs)
int[,.N] # 781 occupation codes

onet_occs <- onet[,.N,by=occ_code][,occ_code]
bls_occs <- bls[,.N,by=occ_code][,occ_code]

bls[,onet_dummy:=fifelse(occ_code %in% onet_occs,1,0)]
onet[,bls_dummy:=fifelse(occ_code %in% bls_occs,1,0)]

# SUMMARY STATS ####
## BLS DATA
bls[onet_dummy==1,.N,by=occ_code][,.N] / bls[,.N,by=occ_code][,.N] # 90% of BLS occ_codes in ONET
bls[,.(tot_emp=sum(tot_emp,na.rm=TRUE)),by=.(onet_dummy,year)][,share:=tot_emp/sum(tot_emp,na.rm=TRUE),by=year] %>% print()
# in 2018-2020, matched occupations account for ~93% of yearly employment, jumps to 97% after 2020
bls[,.(tot_wages=sum(tot_wages,na.rm=TRUE)),by=.(onet_dummy,year)][,share:=tot_wages/sum(tot_wages,na.rm=TRUE),by=year] %>% print()
# similar story for wages, 90-92% in 2018-2020, 97-98% after 2020

# number of industries
bls[,`:=`(naics4=substr(naics,1,4),
          naics3=substr(naics,1,3),
          naics2=substr(naics,1,2))]
bls[,.N,by=naics4][,.N] # 265
bls[,.N,by=naics3][,.N] # 96
bls[,.N,by=naics2][,.N] # 24

# occupations per industry
bls[onet_dummy==1,.(ct=uniqueN(occ_code)),by=naics4][,summary(ct)]
bls[onet_dummy==1,.(ct=uniqueN(occ_code)),by=naics3][,summary(ct)]
bls[onet_dummy==1,.(ct=uniqueN(occ_code)),by=naics2][,summary(ct)]

## ONET
onet[bls_dummy==1,.N,by=occ_code][,.N] / onet[,.N,by=occ_code][,.N] # 98% of onet codes covered
onet2 <- onet[bls_dummy==1]

# number of occupations/tasks/DWAS
onet2[,.N,by=occ_code][,.N] # 781 occupations
onet2[,.N,by=task_id][,.N] # 22080 distinct tasks
onet2[,.N,by=dwa_id][,.N] # 2086 DWAs

onet2[,.N,by=.(occ_code,task_id)][,.N] # 22,184 occupation-tasks
onet2[,.N,by=.(occ_code,task_id,dwa_id)][,.N] # 28,182 occupation-task-DWAs

# distinct tasks without DWA
onet2[,missing_dwa:=fifelse(is.na(dwa_id),1,0)]
onet2[,.(missing_dwa=min(missing_dwa)),by=.(task_id)][missing_dwa==1,.N] / onet2[,.N,by=.(task_id)][,.N] # 2.4% of tasks

# distribution of DWAS per occupation-task
onet2[,.(ct=uniqueN(dwa_id)),by=.(occ_code,task_id)][,summary(ct)]
onet2[,.(ct=uniqueN(dwa_id)),by=.(task_id)][,summary(ct)] # trivial difference

# dist of DWAs per occupation
onet2[,.(ct=uniqueN(dwa_id)),by=.(occ_code)][,summary(ct)]

# dist of tasks per occupation
onet2[,.(ct=uniqueN(task_id)),by=.(occ_code)][,summary(ct)]

onet3 <- copy(onet)
onet3 <- onet3[bls_dummy==1]
onet_quarterly <- onet3[,.N,by=.(year,rel,occ_code,occ_title,task_id,task,dwa_id,dwa_title)]
onet_quarterly[,N:=NULL]

onet_yearly <- onet3[,.N,by=.(year,occ_code,occ_title,task_id,task,dwa_id,dwa_title)]
onet_yearly[,N:=NULL]

write_csv(onet_quarterly,file="output/data/onet_occupations_quarterly.csv")
write_csv(onet_yearly,file="output/data/onet_occupations_yearly.csv")

setcolorder(bls,c("year","naics4","naics_title","occ_code","occ_title","tot_emp","tot_wages"))
bls <- bls[,.(year,naics4,naics_title,occ_code,occ_title,tot_emp,tot_wages)]
write_csv(bls,file="output/data/bls_oews_yearly.csv")

## SHARE MISSING
names(onet_yearly)
onet_yearly[is.na(year),.N] / onet_yearly[,.N]
onet_yearly[is.na(occ_code),.N] / onet_yearly[,.N]
onet_yearly[is.na(occ_title),.N] / onet_yearly[,.N]
onet_yearly[is.na(task_id),.N] / onet_yearly[,.N]
onet_yearly[is.na(task),.N] / onet_yearly[,.N]
onet_yearly[is.na(dwa_id),.N] / onet_yearly[,.N] # 3.37%
onet_yearly[is.na(dwa_title),.N] / onet_yearly[,.N] # 3.37%

names(onet_quarterly)
onet_quarterly[is.na(rel),.N] / onet_quarterly[,.N]
onet_quarterly[is.na(occ_code),.N] / onet_quarterly[,.N]
onet_quarterly[is.na(occ_title),.N] / onet_quarterly[,.N]
onet_quarterly[is.na(task_id),.N] / onet_quarterly[,.N]
onet_quarterly[is.na(task),.N] / onet_quarterly[,.N]
onet_quarterly[is.na(dwa_id),.N] / onet_quarterly[,.N] # 3.23%
onet_quarterly[is.na(dwa_title),.N] / onet_quarterly[,.N] # 3.23%

names(bls)
bls[is.na(year),.N] / bls[,.N]
bls[is.na(naics4),.N] / bls[,.N]
bls[is.na(naics_title),.N] / bls[,.N]
bls[is.na(occ_code),.N] / bls[,.N]
bls[is.na(occ_title),.N] / bls[,.N]
bls[is.na(tot_emp) | tot_emp==0,.N] / bls[,.N] # 6.68%
bls[is.na(tot_wages) | tot_wages==0,.N] / bls[,.N] # 8.36%

# TO DO: look at evolutions over time... 
# - do certain industries see more or less occupations over time, how does this affect employment
# - how does number of DWAs/tasks change over time
# - maybe incorporate skill requirements and look at how these develop over time

