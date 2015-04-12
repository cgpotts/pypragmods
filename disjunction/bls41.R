library(ggplot2)

######################################################################
## Code to run a variant of the experiment in
##
## Chemla, Emmanuel. 2013. Apparent Hurford constraint obviations are
## based on scalar implicatures: an argument based on frequency
## counts. Ms., CNRS, ENS, LSCP Paris.
##
## using Google Books counts rather than search result counts.
##
## The x-axis values (predictors) are from
##
## van Tiel, Bob; Emiel van Miltenburg; Natalia Zevakhina; and Bart
## Geurts. 2013. Scalar diversity. Ms., University of Nijmegen.
##
## ---Christopher Potts
##
######################################################################

library(extrafont)
library(grid)
loadfonts()

## Dataset extending Chemla's numbers with the more reliable Google Books (version 2) counts:
dat = read.csv("vantiel-chemla-googlebooks2.csv",1)

## Y-axis values (dependent variable):
dat$ratio = dat$GB2.Freq.X.or.Y / dat$GB2.Freq.X

## Plotting params:
xlab = expression(paste("Probability of" ~  italic(X) ~ "implicating" ~ italic(not), " ", italic(Y), sep=""))
ylab = expression(italic("Count(X or Y) / Count(X)"))
lab.size = 20
axis.size = 16
text.label.size = 5

## Text to display atop the data points:
labels = paste(dat$X, 'or', dat$Y)
## We display this subset, chosen by hand just to avoid clutter; all
## data points are included in the analysis:
fordisplay = c(1,2,3,4,5,7,8,9,10,14,15,18,19,21,22,26,28,35,42)
sizes = rep(NA,length(labels))
sizes[fordisplay] = text.label.size

## The plot:
ggplot(dat,aes(x=p/100,y=ratio)) +
theme_bw() + 
theme(text=element_text(family="CM Roman", face="bold"), axis.ticks=element_blank(), axis.ticks.length=unit(0,'cm')) +
geom_point(color='#990000',size=5) +
geom_text(label=labels, size=sizes, angle=65, family="CM Roman", face="bold") +
stat_smooth(method="lm") + xlab(xlab) + ylab(ylab) +
coord_cartesian(ylim = c(-0.0003, 0.0015)) + ## Axis that hides one outlier that makes the plot too messy (not excluded from analysis!)
theme(axis.title.x = element_text(face="bold", colour="#000000", size=lab.size, vjust=-0.75), axis.text.x=element_text(size=axis.size, vjust=-1.0)) +
theme(axis.title.y = element_text(face="bold", colour="#000000", size=lab.size, vjust=1.25), axis.text.y=element_text(size=axis.size, hjust=-1.0))

## Save the plot:
outputFilename = "disjunction-and-implicature.pdf"
ggsave(file=outputFilename, width=11, height=8)
embed_fonts(outputFilename, outfile=outputFilename)

# Simple linear model:
fit = lm(ratio ~ p,dat)

print(summary(fit))

