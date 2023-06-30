#!/usr/bin/env Rscript

library(JJHmisc)
library(ggplot2)
library(dplyr)
library(stargazer)

df <- read.csv("../computed_objects/simulation_results.csv") %>%
    mutate(avg_cost_per_task = avg_cost / n) %>%
    mutate(frac_human = avg_human_count / n) %>%
    mutate(fully_automated = frac_human == 0)


m1 <- lm(avg_cost_per_task ~ n, data = df)
m2 <- lm(frac_human ~ cm, data = df)
m3 <- lm(frac_human ~ qmin, data = df)

out.file <- "../writeup/tables/simulation_results.tex"

sink("/dev/null")

s <- stargazer(m1, m2, m3,
               dep.var.labels = c( "Avg. cost/task", "Frac tasks by human"),
               title = "Effect of costs and quality on outputs",
               label = "tab:outcomes",
              # column.separate = c(2,2,2),
               covariate.labels = c("Num. tasks", "Management Cost", "AI model floor (qmin)"),
               omit.stat = c("adj.rsq", "ser", "f"),
               no.space = TRUE,
               star.cutoffs = c(0.10, 0.05, 0.01),
               star.char = c( "*", "**", "***"),
               font.size = "footnotesize",
               column.sep.width = "-6pt",
               #column.labels = c("Event Study", "DD Timing", "DD Pricing"),
               #covariate.labels = c("\\textsc{Post}", "Price, $p$", "\\textsc{Post} $\\times$ $p$"), 
               #add.lines = list(c("Clustered SEs", "", "X", "", "X", "", "X")), 
               header = FALSE,
               type = "latex")

sink()
note <- c("\\\\",
          "\\begin{minipage}{ \\textwidth}",
          "{\\footnotesize \\emph{Notes}: Some notes.}",
          "\\end{minipage}")
JJHmisc::AddTableNote(s, out.file, note)

#stargazer::stargazer(m1, m2, m3)

g <- ggplot(data = df %>% filter(cm %in% c(0.1, 0.5, 1, 1.5) | cm > 1.8), aes(x = frac_human)) +
    geom_histogram() +
    facet_grid(qmin ~ cm) +
    xlab("Model quality, qmin") +
    ylab("Management costs, cm")

JJHmisc::writeImage(g, "fraction_of_tasks_done_by_human", path = "../writeup/plots/")



print(g)

m <- lm(fully_automated ~ qmin, data = df)

ggplot(data = df, aes(x = qmin, y = frac_human, colour = cmin)) +
geom_point() +
facet_wrap(~cm, ncol = 3) +
geom_smooth()


ggplot(data = df, aes(x = avg_human_count, y = avg_cost)) + 
 geom_point()

# As management costs increase, human tasks increase
ggplot(data = df, aes(x = cm, y = avg_human_count)) + 
geom_point()

ggplot(data = df, aes(x = cmin, y = cm, colour = avg_cost)) +  geom_tile() + facet_wrap(~n, ncol = 2)

ggplot(data = df, aes(x = avg_cost)) + facet_wrap(~n, ncol = 3) +  geom_histogram()

ggplot(data = subset(df, n == 10), 
aes(x = avg_cost, colour = avg_human_count)) + 
facet_wrap(~qmin, ncol = 3) + 
geom_histogram()

ggplot(data = subset(df, n == 10), 
aes(x = avg_human_count)) + 
facet_wrap(~qmin, ncol = 3) + 
geom_histogram()


m <- lm(log(avg_cost) ~ log(cmin), data = df)
m <- lm(log(avg_cost) ~ log(cm), data = df)
m <- lm(log(avg_cost) ~ log(cm)*log(qmin), data = df)

m <- lm(std_cost ~ n, data = df)

m <- lm(avg_cost ~ n + qmin + cmin + cm, data = df)
summary(m)

m <- lm(log(avg_cost) ~ log(n) + log(qmin) + log(cmin) + log(cm), data = df)
summary(m)

m <- lm(log(avg_cost) ~ log(n) + log(qmin) + log(cmin)*log(cm), data = df)
summary(m)

# interesting
# As quality gets higher, management costs 
# have non-convex effect on average cost

g <- ggplot(data = df, aes(x = cm, y = avg_cost)) + geom_point() + 
    facet_wrap(~qmin, ncol = 3) +
    xlab("Management cost") +
    ylab("Average cost") +
    theme_bw()

JJHmisc::writeImage(g, "avg_cost_by_management_cost", path = "../writeup/plots/")


ggplot(data = df, aes(x = n, y = avg_cost)) + geom_point()


g <- ggplot(data = df, aes(x = cmin, y = avg_cost)) + 
    facet_wrap(~n, ncol = 2) + 
    geom_point()

JJHmisc::writeImage(g, "example", path = "../writeup/plots/")
