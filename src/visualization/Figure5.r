#install.packages("tidyverse")
library(Rmisc)
library(ggplot2)
library(dplyr)
args = commandArgs(trailingOnly=TRUE) 


# test if there is at least one argument: if not, return an error
if (length(args)==0) {
        stop("At least one argument must be supplied (input file).n", call.=FALSE)
} else if (length(args)==1) {
        # default output file
        args[2] = "out.png"
}

nome = ( paste(args, collapse = " "))


setwd("~/Project_n/Across-Modalities/data/processed/")

df <- read.csv(file = args[1])

tgc <- df %>% 
  group_by(Real.Bin,Modality,Exposures) %>%
  summarise(median = median(Predicted.Bin), percentil25 = quantile(Predicted.Bin, 0.25), percentil75 = quantile(Predicted.Bin, 0.75))

p<- ggplot(tgc, aes(x=Real.Bin, y=Predicted.Bin, group=Exposures, color=Modality)) +   
     facet_wrap(c("Exposures","Modality")) +
     scale_x_continuous(breaks=c(1,2,3,4,5,6)) +
     scale_y_continuous(breaks =c(1,2,3,4,5,6)) +
     geom_line() +
     geom_point()+
     theme_classic()+
     geom_errorbar(aes(ymin=percentil25, ymax=percentil75), width=.2, position=position_dodge(0.05))

p <- p + geom_point(data=df,aes(x=Real.Bin, y=Predicted.Bin, group=Exposures, color=Modality) ) + facet_wrap(c("Exposures","Modality")) 

setwd("~/Project_n/Across-Modalities/reports/figures/")

png(args[2], width = 1000, height =1000)
plot(p)
dev.off()

