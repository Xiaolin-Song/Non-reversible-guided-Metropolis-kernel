Rcpp::sourceCpp('drift-mu3.cpp')
set.seed(123)
library(yuima)
d=50
nu=20
fs<-function(k){
  z1=paste("b",c(1:d),sep="")
  z2=paste("x",c(1:d),sep="")
  z3=paste(z1,"+",z2,sep="")
  z22=z3[1]
  for(i in 2:d){
    z22=paste(z22,",",z3[i],sep="")
  }
  z22=paste("c(",z22,")",sep="")
  z4=paste("-(nu+d)*sum(",z22,"*","bm[,",k,"]",")",sep="")
  z5=paste("(nu+sum(",z22,"%*%","bm","%*%",z22,"))",sep="")
  zp=paste(z4,"/",z5,sep="")
  return(zp)
}
sol <-paste("x",c(1:d),sep="")
sol

b<-matrix("0",d,d)
diag(b)="2^0.5"

a<-c()
for(i in 1:d){
  a[i]=fs(i)
}
bb=rnorm(d)*(10^0.5)
bb=sort(bb)
ll<-list()
for(i in 1:d){
  z=paste("b",i,sep="")
  ll[[z]]=bb[i]
}
true.parameters <- ll
true.parameters

nd<-length(sol)
zz=rWishart(1,50,diag(1,nd,nd))
zz
eigen(zz[,,1])$value
bat=zz[,,1]
bat<-diag(1,d,d)
eigen(bat)$value
bm=solve(bat)
n <- 10^3
T=10
ysamp <- setSampling(Terminal = T, n = n)
ymodel <- setModel(drift = a, diffusion = b, solve.variable = sol)

#true.parameters <- list(mu1=2,mu2=1,mu3=0,mu4=1,mu5=2)
yuima <- setYuima(model = ymodel, sampling = ysamp)
yuima <- simulate(yuima,xinit=0,true.parameter = true.parameters)

plot(yuima@data)
true.parameters

x<-yuima@data
xx<-as.matrix(x@original.data)
dt=T/n
delxx<-xx[2:(n+1),]-xx[1:n,]
dim(delxx)
plot(delxx[,1],type="l")


################################################

##############################stan
code<-"
data {
  int N;
  int D;
  real nu;
  real dt;
  matrix[N,D] delxx;
  matrix[N+1,D] x;
  matrix[D, D] invsig;
  row_vector[D] mu; 
  matrix[D, D] sigma;
}
parameters {
    row_vector[D] beta;
}
model {
   row_vector[D] xt;
   real K;
   to_vector(beta) ~ normal(0,sqrt(10));
   for (i in 1:N){
   xt=row(x,i)+beta;
   K=quad_form(invsig,xt');
   K=-(nu+D)/(nu+K);
   xt=K*(xt*invsig);
   xt=row(delxx,i)-xt*dt;
   xt ~ multi_normal(mu,sigma);
   }
}
"

library(rstan)
fstan<-function(N,wn){
  res<-c()
  warm=wn
  Nn=N
  PD<- stan(model_code=code, 
            data = list(N =n,D=d,nu=nu,dt=dt,delxx=delxx,x=xx,invsig=bm,mu=rep(0,d),sigma=2.0*dt*diag(1,d,d)), 
            chains = 1,             # number of Markov chains
            warmup = warm,          # number of warmup iterations per chain
            iter = Nn,            # total number of iterations per chain
            cores = 1,seed = 1008)
  fit_ss <- extract(PD, permuted =FALSE,inc_warmup =TRUE) # fit_ss is a list 
  zm<-fit_ss[,1,]
  lk=zm[,"lp__"]
  lk<-as.vector(lk)
  beta=fit_ss["beta"][[1]]
  res[[1]]=lk
  res[[2]]=zm[,1:3]
  res[[3]]=get_elapsed_time(PD)[2]
  res[[4]]=summary(PD)$summary[,"n_eff"]["lp__"]
  return(res)
}
############################################################
library(coda)

# set seed
sed=1
for(k in 1:50){
  ## We suggest to run the code in parallel, sice it cost days to run for one iteration.
  sed=k
  set.seed(sed)
  t1<-Sys.time()
  N=1e5# iterations
  res11=fr(N,g=1.2,xx,delxx,dt,bm)# random walk
  res21=fpcn(N,rho=0.95,xx,delxx,dt,ps=1,bm) #pcn
  res31=fmpcn(N,rho=0.01,xx,delxx,dt,ps=1,bm) #mpcn
  res41=fgmpcn(N,rho=0.04,xx,delxx,dt,ps=1,bm) #gmpcn
  res51=fmala(N,xx,delxx,dt,ep=0.15,bm)## mala
  res61=fpcn_mala(N,delta=0.09,xx,delxx,dt,ps=1,bm)## pcn mala
  res71=fpcn_margin(N,delta=1.8,xx,delxx,dt,ps=1,bm) ##  pcn marginal
  res81<-fstan(N,bk)                                ##hmc(stan)
  res91=fsplit_hmc(N,xx,delxx,dt,L=1,ep=0.6,ps=1,bm)## split hmc
  t2<-Sys.time()
  t2-t1
  ll<-c()
  for(i in 1:9){
    ll[[i]]<-get(paste("res",i,"1",sep=""))
  }
  names(ll)<-c("RWM","PCN","MPCN","GMPCN","MALA",
               expression(paste(infinity,"-MALA",)),
               "MGRAD","HMC",
               expression(paste(infinity,"-HMC")))
  
  zz=assign(paste("out",sed,sep=""),ll)
  save(zz,file=paste("out",sed,".Rdata",sep=""))
}


################################################
ll<-c()
for(i in xx){
  load(paste("out",xx[i],".Rdata",sep=""))
  ll[[i]]=kf(zz)
  i=i+1
}

xy<-c(1:50)
tt<-rep(0,9)
for(k in xy){
  ak=ll[[k]]
  tt=ak[,2]+tt
}
tt=tt/length(xy)



mm=matrix(0,nm,9)
for(i in 1:nm){
  ak=ll[[i]]
  mm[i,]=ak[,1]/tt
}
library(ggplot2)
library(reshape2)
dim(mm)
colnames(mm)<-c("RWM","PCN","MPCN","GMPCN","MALA",
                expression(paste(infinity,"-MALA",)),"MGRAD",
                "Stan",expression(paste(infinity,"-HMC")))
data<-as.data.frame(mm)
da<-melt(data)
da$variable<-factor(da$variable,labels =c("RWM","PCN","MPCN","GMPCN","MALA",
                                          expression(paste(infinity,"-MALA",)),"MGRAD",
                                          "Stan",expression(paste(infinity,"-HMC")))
)

##Fig1
ggplot(data=da,aes(x=variable,y=value))+geom_boxplot()+
  theme(legend.position = "none")+
  coord_trans(y="log")+
  scale_y_continuous(breaks=c(0.01,0.1,1,5,10))+
  scale_x_discrete(labels=c("RWM","PCN","MPCN","GMPCN","MALA",
                            expression(paste(infinity,"-MALA",)),"MGRAD",
                            "HMC",expression(paste(infinity,"-HMC"))))+
  ggtitle("ESS of log-likelihood per second")+
  theme(plot.title = element_text(hjust = 0.5))
################################################################
##Fig2 

par(mfrow=c(3,3))
pname<-c("RWM","PCN","MPCN","GMPCN","MALA",
         expression(paste(infinity,"-MALA",)),"MGRAD",
         "Stan",expression(paste(infinity,"-HMC")))
for(i in 1:9){
  lk<-zz[[i]][[1]]
  plot(lk[(N-1e4):N],type="l",ylab = "value",xlab="T",main=pname[i],ylim=c(-25120,-25020))
  }

