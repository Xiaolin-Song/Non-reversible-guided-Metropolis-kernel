#Rcpp::sourceCpp('/home/xiaolin/seminar/Poission-mix/poimix.cpp')
Rcpp::sourceCpp('/home/xiaolin/seminar/Poission-mix/hirach.cpp')
set.seed(1234)
z=1/20
alpha=rgamma(1,z,z)
alpha=1.0
beta=rgamma(1,2,0.2)
beta=3.0
ngroup=25
nob=5
sa=rgamma(ngroup,shape =alpha,rate=beta)
sa
mean(sa)
xx=matrix(0,ngroup,nob)
for(i in 1:ngroup){
  xx[i,]=rpois(nob,sa[i])
}
xx


N=5e4;
rho=0.95;
res1=fcsk(N,xx,rho)
z1<-res1[[2]][,ngroup+1]
plot(z1[100:N],type="l")
accept(res1[[2]][,(ngroup+1):(ngroup+2)])
colMeans(res1[[2]])

res2=fcsk_mix(N,xx,rho)
z2<-res2[[2]][,ngroup+1]
plot(z2[100:N],type="l")
accept(res2[[2]][,(ngroup+1):(ngroup+2)])
colMeans(res2[[2]])

res3=fcsk_guided(N,xx,rho)
z3<-res3[[2]][,ngroup+1]
plot(z3[100:N],type="l")
accept(res3[[2]][,(ngroup+1):(ngroup+2)])
colMeans(res3[[2]])


rho=0.7;
alpha=1.0
res4=fbetag(N,xx,rho,alpha)
z4<-res4[[2]][,ngroup+1]
plot(z4[100:N],type="l")
accept(res4[[2]][,(ngroup+1):(ngroup+2)])
colMeans(res4[[2]])

# 
# alpha=5.0
#rho=0.8;
res5=fbetag_mix(N,xx,rho,alpha)
z5<-res5[[2]][,ngroup+1]
plot(z5[100:N],type="l")
accept(res5[[2]][,(ngroup+1):(ngroup+2)])
colMeans(res5[[2]])


res6=fbetag_guided(N,xx,rho,alpha)
z6<-res6[[2]][,ngroup+1]
plot(z6[100:N],type="l")
accept(res6[[2]][,(ngroup+1):(ngroup+2)])
colMeans(res6[[2]])


ba=1.0/20.0;
be=1.0/20.0;
baa=1.0/20.0;
bee=1.0/20.0;
library(ggplot2)
lik<-function(pa){
  alpha=pa[1];beta=pa[2];
  M=dim(xx)[1];N=dim(xx)[2];
  z=alpha^(ba-1)*exp(-be*alpha)*beta^(baa-1)*exp(-bee*beta);
  for(i in 1:M){
    z=z*gamma(sum(xx[i,])+alpha)/(N+beta)^(sum(xx[i,])+alpha)*beta^alpha/gamma(alpha)
  }
  lik=z;
  return(lik)
}
lik(runif(2))
M=200
upx=3
upy=6
x.points <- seq(0.1,upx,length.out=M)
y.points <- seq(0.1,upy,length.out=M)
data.grid <- expand.grid(x.points,y.points)
q.samp <- cbind(data.grid, prob =lik(data.grid))
colnames(q.samp)<-c("x1","x2","prob")#,"Direction")
q.samp<-as.data.frame(q.samp)
RN = quantile(q.samp[,3],prob=c(0.5,1))
levels=exp(pretty(log(RN),15))


pname<-c("MH with Chi-squared", "MHH with Chi-squared","GMH with Chi-squared", "MH with Beta-Gamma ",
         "MHH with Beta-Gamma ","GMH with Beta-Gamma ")

library(ggplot2)
dl=300
for(i in 1:6){
  ress<-get(paste("res",i,sep=""));
  res<-matrix(0,dl,4)
  res[,1:2]<-ress[[2]][(N-dl+1):N,c(ngroup+1,ngroup+2)]
  res[,3]<-rep(0,dl)
  plot(res[,1],res[,2],type="b")
  if(i==3|i==6){
    res[,4]<-ress[[4]][(N-dl+1):N]
    colnames(res)<-c("x1","x2","prob","Direction")
    res<-as.data.frame(res)
    res$Direction<-as.factor(res$Direction)
    mytype=c(1,2)
    p=ggplot(q.samp,aes(x=x1,y=x2,z=prob))+ 
      stat_contour(breaks=levels,col=1,size=0.1)+
      geom_segment(data=res,aes(xend = lead(x1),yend = lead(x2),lty=Direction))+
      theme_bw()+
      ggtitle(pname[i])+
      theme(axis.line = element_line(colour ="black"),
            legend.title=element_text(size=8),
            legend.text=element_text(size=8),
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(),
            panel.border = element_blank(),
            panel.background = element_blank(),
            #legend.position = c(12,6),
            legend.position = c(0.9,0.2),
            plot.title = element_text(hjust = 0.5))+
      scale_y_continuous(limits = c(0.1,upy))+
      scale_x_continuous(limits = c(0.1,upx))+
      labs(x=expression(alpha),y=expression(beta))
    assign(paste("p_",i,sep=""),p)
  }else{
    colnames(res)<-c("x1","x2","prob","d")
    res<-as.data.frame(res)
    p=ggplot(q.samp, aes(x=x1, y=x2, z=prob))+ 
      stat_contour(breaks=levels,col=1,size=0.1)+
      geom_path(data=res,col=1)+
      scale_y_continuous(limits = c(0.1,upy))+
      scale_x_continuous(limits = c(0.1,upx))+
      theme_bw()+
      ggtitle(pname[i])+
      theme(axis.line = element_line(colour ="black"),
            legend.title=element_text(size=8),
            legend.text=element_text(size=8),
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(),
            panel.border = element_blank(),
            panel.background = element_blank(),
            #legend.position = c(12,6),
            legend.position = c(0.9,0.2),
            plot.title = element_text(hjust = 0.5))+
      labs(x=expression(alpha),y=expression(beta))
    
    assign(paste("p_",i,sep=""),p)
  }
}
library(gridExtra)
library(dplyr)

pl=grid.arrange(p_1,p_2,p_3,p_4,p_5,p_6,nrow=2)
pl
