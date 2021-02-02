
library(mlbench)
library(mcmcse)
data=read.csv('C:/Users/song/Desktop/onarnew/sonar', header=FALSE)
dim(data)
data<-data[,1:61]
#data<-as.matrix(data)
dim(data)
XX<-data[,1:60]
YY<-data[,61]


YY<-as.numeric(YY)
YY<-YY-1
YY
dd=length(YY)
X<-as.matrix(XX)
Y<-YY


setwd(('C:/Users/song/Desktop/onarnew/'))
Rcpp::sourceCpp('tun-mu-sigma-old.cpp')
library(mlbench)

library(coda)
#############################################################


load("lt.Rdata")
sigk<-lt[[1]]
vvk<-lt[[2]]

set.seed(135)
N=1e4
res_1<-frr(X,Y,N,1.0,sigk)
res_2<-fpcn(X,Y,N,0.3,sigk,vvk)
res_3<-fmpcn(X,Y,N,0.45,sigk,vvk)
res_4<-fgmpcn(X,Y,N,0.45,sigk,vvk)
res_5<-fmala(X,Y,N,0.7,sigk,vvk)
res_6<-fpcnmala(X,Y,N,0.55,sigk,vvk)
res_7<-fpcnmargin(X,Y,N,2.5,sigk,vvk)
res_8=fsplit_hmc(X,Y,N,1.0,1,sigk,vvk)
par(mfrow=c(3,3))
for(i in 1:8){
  res=get(paste("res_",i,sep=""))
  plot(res[[1]][1:1e4],type="l")
}

code<-"
data {
  int N;
  int D;
  matrix[N,D] x;
  int<lower=0,upper=1> y[N];
}
parameters {
    vector[D] beta;
}
model {
   vector[N] x_beta = x * beta;
   to_vector(beta) ~ normal(0, sqrt(100));
   for (i in 1:N){
     y[i] ~ bernoulli_logit(x_beta[i]);
   }
}
"
library(rstan)
dd=dim(X)[2]
d1=dim(X)[1]

fstan<-function(nn,wn,Y,X){
  t1<-Sys.time()
  t1<-as.numeric(t1)
  res_9<-c()
  warm=wn
  Nn=nn+warm
  PD<- stan(model_code=code, 
            data = list(N =208,D=dd,y=Y,x=X),
            chains = 1,             # number of Markov chains
            warmup = warm,          # number of warmup iterations per chain
            iter = Nn,            # total number of iterations per chain
            cores = 1,seed = t1)
  
  fit_ss <- extract(PD, permuted =FALSE,inc_warmup =TRUE) # fit_ss is a list 
  zm<-fit_ss[,1,]
  lk=zm[,"lp__"]
  lk<-as.vector(lk)
  beta=fit_ss["beta"][[1]]
  res_9[[1]]=lk
  res_9[[2]]=zm[,1:3]
  res_9[[3]]=summary(PD)$summary[,"n_eff"]["lp__"]
  return(res_9)
}

res_9=fstan(8000,2000,Y,X)
par(mfrow=c(3,3))
for(i in 1:8){
  res=get(paste("res_",i,sep=""))
  plot(res[[1]][1:2e4],type="l")
}

max(res_1[[1]][1:1e4])

N=1e4
res11<-res1
library(ggplot2)
nm=1
#kf(c(1,2,3,4,5,6,7,8))
for(i in c(1:8)){
  pnames=c("RWM","PCN","MPCN","GMPCN","MALA",
           expression(paste(infinity,"-MALA",)),
           "MGRAD",
           expression(paste(infinity,"-HMC")),"STAN")
  res=get(paste("res_",i,sep=""))
  dat=cbind(nm:N,res[[1]][nm:N])
  dat<-as.data.frame(dat)
  colnames(dat)=c("iteration","value")
  p=ggplot(dat,aes(x=iteration,y=value))+ 
    geom_line()+
    ggtitle(pnames[i]) +
    scale_x_continuous(breaks=c(1,5000,10000))+ 
    #scale_y_continuous(breaks=c(-200,-140,-80))+
    scale_y_continuous(breaks=c(-300,-100),limits=c(-300,0.1))+
    theme(axis.line = element_line(colour ="black"),
          legend.title=element_text(size=8),
          legend.text=element_text(size=8),
          plot.title = element_text(hjust = 0.5))
  assign(paste("pp_",i,sep=""),p)
}
library(gridExtra)
pl=grid.arrange(pp_1,pp_3,pp_4,pp_5,pp_6,pp_7,nrow=2,ncol=3)
pl
