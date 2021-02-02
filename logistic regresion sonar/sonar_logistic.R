
setwd(('C:/Users/song/Desktop/onarnew/github-sonar'))
library(mlbench)
library(mcmcse)
data=read.csv('sonar', header=FALSE)
dim(data)
data<-data[,1:61]
dim(data)
XX<-data[,1:60]
YY<-data[,61]
YY<-as.numeric(YY)
YY<-YY-1
YY
dd=length(YY)
X<-as.matrix(XX)
Y<-YY


Rcpp::sourceCpp('logistic.cpp')
library(mlbench)
set.seed(1237)
library(coda)
library(mcmcse)
#######################################################################

###calculate ess based on coda package

ff<-function(res){
  N=length(res[[1]])
  nk=2e4
  mm<-c()
  lik=res[[1]][nk:N]
  k1<-effectiveSize(lik)
  #k1=multiESS(res[[2]][nk:N,])
  t3=res[[3]]*(N-nk)/N
  mm[1]<-k1
  mm[2]<-t3
  mm[3]<-k1/(t3)
  mm[4]<-accept(res[[2]][(N-nk):N,])
  return(mm)
}

kf<-function(z){
  dl=length(z)
  mm<-matrix(0,dl,4)
  colnames(mm)<-c("ESSL","time-cost",paste("ESSL/second",dd,sep=""),
                  "acceptance_rate")
  pnames=c("RWM","PCN","MPCN","GMPCN","MALA",
           expression(paste(infinity,"-MALA",)),
           "MGRAD","HMC",
           expression(paste(infinity,"-HMC")))
  rownames(mm)<-pnames[1:dl]
  for(i in 1:dl){
    res=get(paste("res_",z[i],sep=""))
    mm[i,]<-ff(res)
  }
  return(round(mm,2))
}


###############################################################precondition /mu /Sigma

nt=2e5
rest<-fr(X,Y,nt,0.4,pt=0)
plot(rest[[1]][1e5:nt],type="l")
lt<-c()
lt[[1]]<-cov(rest[[2]][1e4:nt,])
lt[[2]]<-colMeans(rest[[2]][1e4:nt,])
# save(lt,file="lt.Rdata")
# load("lt.Rdata")
sigk<-lt[[1]] ##precondition /mu
vvk<-lt[[2]]  ##precondition /Sigma


dd<-dim(X)[2]
sigkt<-diag(1,dd,dd)
sigko<-diag(100,dd,dd)
vvkno<-rep(0,dd)
################################################################### Stan code

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
  res_9[[3]]=get_elapsed_time(PD)[2]
  #res_9[[3]]=summary(PD)$summary[,"n_eff"]["lp__"]
  return(res_9)
}




#################################################


#tuning ep for infinity-hmc
for(i in 1:10){
  zz=fsplit_hmc(X,Y,5e4,(0.1*i+0.1),2,sigk,vvk)
  plot(zz[[1]][1000:5e4],type="l")
  print(ff(zz))
}

for(i in 1:5){
  zz=fsplit_hmc(X,Y,5e4,1.0,i,sigk,vvk)
  ff(zz)
  print(ff(zz))
  plot(zz[[1]][10:5e4],type="l")
}






N=1e5
ll<-c()
for(i in 1:1){
  t1<-Sys.time()
  res_1<-frr(X,Y,N,1.0,sigk)
  res_2<-fpcn(X,Y,N,0.3,sigk,vvk)
  res_3<-fmpcn(X,Y,N,0.45,sigk,vvk)
  res_4<-fgmpcn(X,Y,N,0.45,sigk,vvk)
  res_5<-fmala(X,Y,N,0.54,sigk,vvk)
  res_6<-fpcnmala(X,Y,N,0.53,sigk,vvk)
  res_7<-fpcnmargin(X,Y,N,6.5,sigk,vvk)
  res_8<-fsplit_hmc(X,Y,N,1.0,1,sigk,vvk)
  res_9<-fstan(1000,200,Y,X)
  zz<-kf(c(1,2,3,4,5,6,7,8,9))
  t2<-Sys.time()
  print(t2-t1)
  print(zz)
  ll[[i]]<-zz
  print(i)
}
ll






mm<-matrix(0,9,50)
for(i in 1:50){
  zz<-ll[[i]]
  zh<-lh[[i]][2]
  mm[,i]=c(zz[,3],zh)
}
rownames(mm)<-c("RWM","PCN","MPCN","GMPCN","MALA",
                expression(paste(infinity,"-MALA",)),"MGRAD",
                expression(paste(infinity,"HMC")),"Stan")
mm
boxplot(t(mm))



library(ggplot2)
library(reshape2)
data<-as.data.frame(t(mm))
da<-melt(data)
da$variable<-factor(da$variable,labels =c("RWM","PCN","MPCN","GMPCN","MALA",
                                          expression(paste(infinity,"-MALA",)),"MGRAD",
                                          expression(paste(infinity,"HMC")),"Stan") )
ggplot(data=da,aes(x=variable,y=value))+geom_boxplot()+
  theme(legend.position = "none")+
  scale_x_discrete(labels=c("RWM","PCN","MPCN","GMPCN","MALA",
                            expression(paste(infinity,"-MALA",)),"MGRAD",
                            expression(paste(infinity,"HMC")),"HMC"))
