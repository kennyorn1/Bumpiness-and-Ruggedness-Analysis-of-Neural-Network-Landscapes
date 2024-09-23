library(ggplot2)
library(dplyr)
library(plotly)
library(tidyr)
library(reshape2)

result_xor=minist
colnames(result_xor)<-c('epoch','loss','accuracy','ruggedness','bumpiness','avgTrace','MLP')
datas=result_xor
epoch=datas[,1]
loss=datas[,2]
accuracy=datas[,3]
ruggedness=datas[,4]
bumpiness=datas[,5]
avgTrace=datas[,6]
MLP=datas[,7]
sample1=ggplot(datas,aes(x=epoch,y=ruggedness,group=MLP,color=MLP))+
  geom_line()+
  xlab("epoch")+
  ylab("ruggedness")+
  theme_bw()
sample1
ggsave('ruggedness.png',sample1,dpi = 150)
sample1=ggplot(datas,aes(x=epoch,y=bumpiness,group=MLP,color=MLP))+
  geom_line()+
  xlab("epoch")+
  ylab("bumpiness")+
  theme_bw()
sample1
ggsave('bumpiness.png',sample1,dpi = 150)
sample1=ggplot(datas,aes(x=epoch,y=avgTrace,group=MLP,color=MLP))+
  geom_line()+
  xlab("epoch")+
  ylab("avgTrace")+
  theme_bw()
sample1
ggsave('avgTrace.png',sample1,dpi = 150)
result_xor <- melt(result_xor,id=c("epoch","MLP"))
datas=result_xor
##########################3d##########################

MLP=datas$MLP
sample1=ggplot(datas,aes(x=epoch,y=value,group=variable,color=variable))+
  geom_line()+
  xlab("epoch")+
  ylab("value")+
  theme_bw()+
  facet_wrap(.~MLP)
sample1
ggsave('whole.png',sample1,dpi = 150)

