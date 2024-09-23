library(ggplot2)
library(dplyr)
library(plotly)
library(tidyr)

result_xor=result_diabetes3
colnames(result_xor)<-c('loss','gradient','type','residual','step','seed','ruggedness', 'bumpiness')

##########################3d##########################
datas=result_xor
datas[,8]=datas[,8]/datas[,7]
loss=datas[,1]
gradient=datas[,2]
ruggedness=datas[,7]
bumpiness=datas[,8]

sample1=ggplot(datas,aes(x=step,y=bumpiness))+
  geom_point(aes(colour=bumpiness),alpha=1)+
  facet_wrap(.~residual)+
  scale_color_gradientn(colours =rainbow(10))
sample1

sample1=ggplot(datas,aes(group=step,y=bumpiness))+
  geom_boxplot()
sample1

# sample1=ggplot(datas,aes(x=step,y=ruggedness))+
#   geom_line(aes(colour=bumpiness),alpha=1)+
#   facet_wrap(.~residual)+
#   scale_color_gradientn(colours =rainbow(10))
# sample1
