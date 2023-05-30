#!/usr/bin/env Rscript

library(JJHmisc)
library(ggplot2)
library(dplyr)

df <- read.csv("../computed_objects/simulation_results.csv") %>%
    mutate(avg_cost_per_task = avg_cost / n) %>%
    mutate(frac_human = avg_human_count / n) %>%
    mutate(fully_automated = frac_human == 0)

m <- lm(avg_cost_per_task ~ n, data = df)

# high management costs, more human-done tasks
m <- lm(frac_human ~ cm, data = df)

# higher AI quality, fewer human-done tasks
m <- lm(frac_human ~ qmin, data = df)

ggplot(data = df, aes(x = frac_human)) + 
geom_histogram()

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

ggplot(data = df, aes(x = cmin, y = cm, colour = avg_cost)) + 
geom_tile() + facet_wrap(~n, ncol = 2)

ggplot(data = df, aes(x = avg_cost)) + facet_wrap(~n, ncol = 3) + 
geom_histogram()

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
ggplot(data = df, aes(x = cm, y = avg_cost)) + geom_point() + 
facet_wrap(~qmin, ncol = 2)

ggplot(data = df, aes(x = n, y = avg_cost)) + geom_point()


g <- ggplot(data = df, aes(x = cmin, y = avg_cost)) + 
    facet_wrap(~n, ncol = 2) + 
    geom_point()

JJHmisc::writeImage(g, "example", path = "../writeup/plots/")
