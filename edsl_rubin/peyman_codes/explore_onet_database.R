# File: explore_onet_data.R
# Summary: This file explores every component of the O*Net database

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

### load in quarterly occupation data
path <- paste0("input/onet/db_28_2_excel.zip")
outdir <- "input/onet/unzipped"
unzip(zipfile=path,exdir=outdir)

files <- list.files(paste0(outdir,"/db_28_2_excel/"),full.names = TRUE,)

for (file in files) {
  if (file!="input/onet/unzipped/db_28_2_excel/Read Me.txt") {
    print(file)
    dt <- setDT(read_excel(file))
    names(dt) <- fix_names(names(dt))
    print(names(dt))
    print("")
    print("")
    print("")
  }
  gc()
}

file <- "input/onet/unzipped/db_28_2_excel/Work Styles.xlsx"

dt <- setDT(read_excel(file))
names(dt) <- fix_names(names(dt))
print(names(dt))
view(dt)
dt[,.N,by=.(onetsoc_code,commodity_code)][,.N,by=onetsoc_code][,summary(N)]

dt[,.N,by=.(element_id,anchor_value)][,.N]


file <- "input/onet/unzipped/db_28_2_excel/Content Model Reference.xlsx"

dt <- setDT(read_excel(file))
names(dt) <- fix_names(names(dt))

dt[,len:=str_length(gsub("\\.","",element_id))]
dt[,c("cat1","cat2","cat3","cat4","cat5","cat6","cat7"):=tstrsplit(element_id,"\\.")]
dt[!is.na(cat1),cat:="level_1"]
dt[!is.na(cat2),cat:="level_2"]
dt[!is.na(cat3),cat:="level_3"]
dt[!is.na(cat4),cat:="level_4"]
dt[!is.na(cat5),cat:="level_5"]
dt[!is.na(cat6),cat:="level_6"]
dt[!is.na(cat7),cat:="level_7"]

level1 <- dt[cat=="level_1",.(element_name,cat1)]
level2 <- dt[cat=="level_2",.(element_id,element_name,cat1,cat2)]
level3 <- dt[cat=="level_3",.(element_id,element_name,cat1,cat2,cat3)]

list1 <- c()
list2 <- c()
tables <- list()
for (file in files) {
  if (file!="input/onet/unzipped/db_28_2_excel/Read Me.txt") {
    dt <- setDT(read_excel(file))
    names(dt) <- fix_names(names(dt))
    
    if ("element_id" %in% names(dt)) {
      list1 <- c(list1,file)
      
      dt[,name:=paste0(file)]
      dt <- dt[,.N,by=.(name,element_id,element_name)]
      tables[[paste0(file)]] <- dt
      
    } else {
      list2 <- c(list2,file)
    }
  }
  gc()
}

db <- rbindlist(tables)
db[,name:=gsub("input/onet/unzipped/db_28_2_excel/","",name)]
db[,len:=str_length(element_id)]
db[,c("cat1","cat2","cat3","cat4","cat5","cat6","cat7"):=tstrsplit(element_id,"\\.")]
db[!is.na(cat1),cat:="level_1"]
db[!is.na(cat2),cat:="level_2"]
db[!is.na(cat3),cat:="level_3"]
db[!is.na(cat4),cat:="level_4"]
db[!is.na(cat5),cat:="level_5"]
db[!is.na(cat6),cat:="level_6"]
db[!is.na(cat7),cat:="level_7"]

db[cat=="level_1",.N,by=name]
db[cat=="level_2",.N,by=name]
db[cat=="level_3",.N,by=name]
db[cat=="level_4",.N,by=name]
db[cat=="level_5",.N,by=name]
db[cat=="level_6",.N,by=name]
db[cat=="level_7",.N,by=name]

db <- db[name!="Content Model Reference.xlsx"]
lvl1 <- db[cat=="level_1",.(name1=name,cat1)]
lvl2 <- db[cat=="level_2",.(name2=name,cat1,cat2)]
lvl3 <- db[cat=="level_3",.(name3=name,cat1,cat2,cat3)]
lvl4 <- db[cat=="level_4",.(name4=name,cat1,cat2,cat3,cat4)]
lvl5 <- db[cat=="level_5",.(name5=name,cat1,cat2,cat3,cat4,cat5)]
lvl6 <- db[cat=="level_6",.(name6=name,cat1,cat2,cat3,cat4,cat5,cat6)]
lvl7 <- db[cat=="level_7",.(name7=name,cat1,cat2,cat3,cat4,cat5,cat6,cat7)]

db_map <- merge.data.table(lvl1,lvl2,by=c("cat1"),all=TRUE)
db_map <- merge.data.table(db_map,lvl3,by=c("cat1","cat2"),all=TRUE)
db_map <- merge.data.table(db_map,lvl4,by=c("cat1","cat2","cat3"),all=TRUE)
db_map <- merge.data.table(db_map,lvl5,by=c("cat1","cat2","cat3","cat4"),all=TRUE)
db_map <- merge.data.table(db_map,lvl6,by=c("cat1","cat2","cat3","cat4","cat5"),all=TRUE)
db_map <- merge.data.table(db_map,lvl7,by=c("cat1","cat2","cat3","cat4","cat5","cat6"),all=TRUE)

db_map <- db_map[,.N,by=.(name1,name2,name3,name4,name5,name6,name7)]

