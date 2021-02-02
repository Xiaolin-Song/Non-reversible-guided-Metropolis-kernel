#include <RcppArmadillo.h>
#include <Rcpp/Benchmark/Timer.h>
// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends("mcmcse")]]

#include <RcppArmadilloExtensions/sample.h>
using namespace Rcpp;
using namespace arma;

static double const log2pi = std::log(2.0 * M_PI);
const double sdd=100.0; //prior


// [[Rcpp::export]]

int fnn(int m){
  int lik=0;int a;
  for(int i=0;i<300;i++){
    lik +=std::pow(i,2);a=i;
    if(m>-1&&m<=lik){break;}
  }
  return (a);
}

/* accept rate*/
// [[Rcpp::export]]
double accept(arma::mat x){
  int nn=x.n_rows;
  int s=0;
  for(int i=1;i<nn;i++){
    //arma::umat ss=(x.row(i)==x.row(i-1));
    if(x(i,0)==x(i-1,0)){
      s=s;
    }else{
      s=s+1;
    }
  }
  double ac=s/(double)nn;
  return(ac);
}




// log-likelihood
// [[Rcpp::export]]
double loglik(arma::rowvec x,arma::mat xx,arma::rowvec yy){
  double aa=-0.5*arma::sum((x)*x.t())/(sdd);
  int n=yy.n_elem;
  double sigm;
  for(int i=0;i<n;i++){
    double K=arma::sum(-x%xx.row(i));
    double KK=(1.0+std::exp(K));
    if(yy(i)==1){
      aa=-std::log(KK)+aa;
    }else{
      aa=K-std::log(KK)+aa;
    }
  }
  return(aa);
}



//gradient for log-likelihood
// [[Rcpp::export]]
arma::rowvec dif(arma::rowvec x,arma::mat xx,arma::rowvec yy){
  int n=xx.n_rows;
  arma::rowvec dif=-x/sdd;
  for(int i=0;i<n;i++){
    double sigm=1.0/(1.0+std::exp(arma::sum(-x%xx.row(i))));
    dif=(yy(i)-sigm)*xx.row(i)+dif;
  }
  return(dif);
}


// // [[Rcpp::export]]
// arma::rowvec diff(arma::rowvec x,arma::mat xx,arma::rowvec yy,arma::mat invsig){
//   int n=xx.n_rows; 
//   int m=xx.n_cols;arma::rowvec rr(m);rr.zeros();
//   rr=dif(x,xx,yy)+x*invsig;
//   return(rr);
// }

// log-likelihood with precondition-mu
// [[Rcpp::export]]
arma::vec loglikm(arma::rowvec cf,arma::mat xx,arma::rowvec yy,arma::rowvec vv,arma::mat invsig){
  arma::vec zl(2);
  arma::rowvec f=cf+vv;
  double ps=-0.5*arma::sum(cf*invsig*cf.t());
  int n=yy.n_elem;
  double likful=loglik(f,xx,yy);
  zl(0)=likful-ps;
  zl(1)=likful;
  return(zl);
}

// random walk Metopolis kernel
// [[Rcpp::export]]
List rwm(arma::mat x,arma::rowvec Y,int N, arma::mat cholsig, double g,
         arma::rowvec firstrow){
  List out(2);
  int n=x.n_cols;arma::mat res(N,n);res.row(0)=firstrow;
  arma::rowvec rd(n),log_like(N),pro(n);
  res.row(0)=firstrow;
  log_like(0)=loglik(res.row(0),x,Y);
  for(int i=1;i<N;i++){
    rd=rd.randn()*cholsig;
    pro=res.row(i-1)+2.38*rd/(std::pow(n,0.5))*g;
    double a=loglik(pro,x,Y);
    double acc=a-log_like(i-1);
    double u=std::log(arma::as_scalar(arma::randu(1)));
    if(acc>u){
      res.row(i)=pro;
      log_like(i)=a;
    }else{
      res.row(i)=res.row(i-1);
      log_like(i)=log_like(i-1);
    }
  }
  out(0)=log_like;out(1)=res;
  return(out);
}


// adaptive random walk Metopolis kernel
// [[Rcpp::export]]
List fr(arma::mat x,arma::rowvec Y,int N,double g,int pt){
  List rm(2),out(3);
  int n=x.n_cols;
  arma::rowvec rd(n),lk(N),pro(n),av_new(n);
  arma::mat res(N,n);res.zeros();res.row(0)=rd.randn();
  arma::mat sig(n,n);sig=sig.eye();
  arma::mat cholsig=arma::chol(sig);
  
  int M=1e4;
  rm=rwm(x,Y,M,cholsig,g,res.row(0));
  arma::mat bm2=rm(1); arma::rowvec lk2=rm(0);
  double acc=accept(bm2);
  Rcout << "The accept rate" << acc << "\n";
  
  arma::mat covmat=arma::cov(bm2);
  arma::rowvec av=arma::mean(bm2,0);
  arma::mat bpm=covmat*2.38*2.38/std::pow(n,1);
  arma::mat snew(n,n);snew=arma::chol(bpm);
  res.rows(0,M-1)=bm2;lk.subvec(0,M-1)=lk2;
  clock_t start=std::clock();
  
  for(int i=M;i<N;i++){
    arma::rowvec vv_pro(n);vv_pro.randn(); //proposal  random vector
    pro=res.row(i-1)+vv_pro.randn()*snew; //new proposal
    double lik_new=loglik(pro,x,Y);
    double u=arma::as_scalar(arma::randu(1));
    double alpha=lik_new-lk(i-1);
    if(alpha>std::log(u)){
      res.row(i)=pro;
      lk(i)=lik_new;
    }else{
      res.row(i)=res.row(i-1);
      lk(i)=lk(i-1);
    }
    av_new=(i*av+res.row(i))/(i+1);
    arma::mat md1=(i*av.t()*av-(i+1)*(av_new.t()*av_new)+res.row(i).t()*res.row(i));
    covmat=((i-1)*covmat+md1)/i;
    av=av_new;
    
    if(fnn(i-M+1)-fnn(i-M)>0){
      arma::mat bpm=covmat*2.38*2.38/n*0.8;
      snew=arma::chol(bpm);
    }
  }
  
 
  double t4 = (std::clock() - start )*1000/CLOCKS_PER_SEC;
  
  acc=accept(res.rows(M,N-1));
  Rcout << "The accept rate" << acc << "\n";
  
  uvec IDX = regspace<uvec>(0,30,n-1);
  arma::mat ress=res.cols(IDX);
  out(0)=lk;out(1)=res;out(2)=t4/1000.0;//out(3)=sec(res.rows(1e4,N-1));
  return(out);
}

// [[Rcpp::export]]
List frr(arma::mat x,arma::rowvec Y,int N,double g,arma::mat sigk){
  List rm(2),out(3);
  int n=x.n_cols;
  arma::rowvec rd(n),lk(N);
  arma::mat res(N,n);res.zeros();res.row(0)=rd.randn();
  arma::mat sig(n,n);sig=sigk;
  arma::mat cholsig=arma::chol(sig);
  
  int M=1;
  clock_t start=std::clock();
  rm=rwm(x,Y,(N-M),cholsig,g,res.row(0));
  arma::mat bm2=rm(1); arma::rowvec lk2=rm(0);
  res.rows(M,N-1)=bm2;lk.subvec(M,N-1)=lk2;
  double t4 = (std::clock() - start )*1000/CLOCKS_PER_SEC;
  
  double acc=accept(bm2);
  Rcout << "The accept rate" << acc << "\n";
  
  uvec IDX = regspace<uvec>(0,30,n-1);
  arma::mat ress=res.cols(IDX);
  out(0)=lk;out(1)=ress;out(2)=t4/1000.0;//out(3)=sec(res.rows(1e4,N-1));
  return(out);
}



// guided Metropolis-Hastings kernel
// [[Rcpp::export]]
List gmpcn(arma::mat x,arma::rowvec Y,int N, arma::mat sig,arma::rowvec vv, double rho,
           arma::rowvec firstrow){
  List out(2);int n=x.n_cols;
  arma::mat C(n,n);C=arma::chol(sig);
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::mat res(N,n);res.row(0)=firstrow;
  arma::rowvec rd(n),log_like(N),pro(n),pro_t(n),v(N);
  res.row(0)=firstrow;
  log_like(0)=loglik(res.row(0),x,Y);
  double bb,bbn,r1,r2;
  
  r1=std::pow(rho,0.5);r2=std::pow(1.0-rho,0.5);
  arma:: rowvec orgv=(res.row(0)-vv)*invsnew;
  double aa=arma::norm(orgv,2);
  double gg=0;v(0)=1;
  for(int i=1;i<N;i++){
    if(v(i-1)==1){
      for(;;){
        rd=rd.randn();
        gg=R::rgamma(0.5*n,2.0/(aa*aa));
        pro_t=r1*orgv+r2*rd*std::pow(gg,-0.5);
        bb=arma::norm(pro_t,2);
        if(bb<aa){
          break;
        }
      }
    }else{
      for(;;){
        rd=rd.randn();
        gg=R::rgamma(0.5*n,2.0/(aa*aa));
        pro_t=r1*orgv+r2*rd*std::pow(gg,-0.5);
        bb=arma::norm(pro_t,2);
        if(bb>aa){
          break;
        }
      }
    }
    pro=vv+pro_t*C;
    double a=loglik(pro,x,Y);
    double acc=a-log_like(i-1)+n*std::log(bb/aa);
    double u=std::log(arma::as_scalar(arma::randu(1)));
    if(acc>u){
      res.row(i)=pro;
      log_like(i)=a;
      v(i)=v(i-1);
      orgv=pro_t;
      aa=bb;
    }else{
      res.row(i)=res.row(i-1);
      log_like(i)=log_like(i-1);
      v(i)=-1*v(i-1);
    }
    //Rcout << "The accept rate" << i << "\n";
  }
  out(0)=log_like;out(1)=res;
  return(out);
}



// [[Rcpp::export]]
List fgmpcn(arma::mat x,arma::rowvec Y,int N,double rho,arma::mat sigk,arma::rowvec vvk){
  List out(5),rm(2);
  int n=x.n_cols;
  arma::mat sig(n,n);sig=sigk;
  
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::rowvec rd(n),lk(N),pro(n),pro_t(n),v(n),vv(n);vv=vvk;
  arma::mat res(N,n);res.row(0)=rd.randn();
  lk(0)=loglik(res.row(0),x,Y);
  int M=1;
  Rcout << "The rho" << rho << "\n";
  
  clock_t start=std::clock();
  
  rm=gmpcn(x,Y,(N-M),sig,vv,rho,res.row(0));
  arma::mat bm2=rm(1); arma::rowvec lk2=rm(0);
  res.rows(M,N-1)=bm2;lk.subvec(M,N-1)=lk2;
  //res=bm2;lk=lk2;
  uvec IDX = regspace<uvec>(0,30,n-1);
  double t4 = (std::clock() - start )*1000/CLOCKS_PER_SEC;
  double acc=accept(bm2);
  Rcout << "The accept rate" << acc << "\n";
  arma::mat ress=res.cols(IDX);
  out(0)=lk;out(1)=ress;out(2)=t4/1000.0;//out(3)=sec(res.rows(1e4,N-1));
  return(out);
}


//mpcn kernel
// [[Rcpp::export]]
List mpcn(arma::mat x,arma::rowvec Y,int N, arma::mat sig,arma::rowvec vv, double rho,
          arma::rowvec firstrow){
  List out(2);int n=x.n_cols;
  arma::mat C(n,n);C=arma::chol(sig);
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::mat res(N,n);res.row(0)=firstrow;
  arma::rowvec rd(n),log_like(N),pro(n),pro_t(n),v(N);
  log_like(0)=loglik(res.row(0),x,Y);
  double bb,bbn,r1,r2;
  
  r1=std::pow(rho,0.5);r2=std::pow(1.0-rho,0.5);
  arma:: rowvec orgv=(res.row(0)-vv)*invsnew;
  double aa=arma::norm(orgv,2);
  double gg=0;v(0)=1;
  for(int i=1;i<N;i++){
    rd=rd.randn();
    gg=R::rgamma(0.5*n,2.0/(aa*aa));
    pro_t=r1*orgv+r2*rd*std::pow(gg,-0.5);
    bb=arma::norm(pro_t,2);
    pro=vv+pro_t*C;
    double a=loglik(pro,x,Y);
    double acc=a-log_like(i-1)+n*std::log(bb/aa);
    double u=std::log(arma::as_scalar(arma::randu(1)));
    if(acc>u){
      res.row(i)=pro;
      log_like(i)=a;
      orgv=pro_t;
      aa=bb;
    }else{
      res.row(i)=res.row(i-1);
      log_like(i)=log_like(i-1);
    }
    //Rcout << "The accept rate" << i << "\n";
  }
  out(0)=log_like;out(1)=res;
  return(out);
}

// [[Rcpp::export]]
List fmpcn(arma::mat x,arma::rowvec Y,int N,double rho,arma::mat sigk,arma::rowvec vvk){
  List out(5),rm(2);
  int n=x.n_cols;
  arma::mat sig(n,n);sig=sigk;
  
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::rowvec rd(n),lk(N),pro(n),pro_t(n),v(n),vv(n);vv=vvk;
  arma::mat res(N,n);res.row(0)=rd.randn();
  lk(0)=loglik(res.row(0),x,Y);
  int M=1;
  Rcout << "The rho" << rho << "\n";
  
  clock_t start=std::clock();
  
  rm=mpcn(x,Y,(N-M),sig,vv,rho,res.row(0));
  arma::mat bm2=rm(1); arma::rowvec lk2=rm(0);
  res.rows(M,N-1)=bm2;lk.subvec(M,N-1)=lk2;
  //res=bm2;lk=lk2;
  uvec IDX = regspace<uvec>(0,30,n-1);
  double t4 = (std::clock() - start )*1000/CLOCKS_PER_SEC;
  double acc=accept(bm2);
  Rcout << "The accept rate" << acc << "\n";
  arma::mat ress=res.cols(IDX);
  out(0)=lk;out(1)=ress;out(2)=t4/1000.0;//out(3)=sec(res.rows(1e4,N-1));
  return(out);
}


//mala kernel
// [[Rcpp::export]]
List mala(arma::mat x,arma::rowvec Y,int N,arma::mat sig,double ep,arma::rowvec fstrow){
  List out(2);
  int n=x.n_cols;
  arma::mat C(n,n);C=arma::chol(sig);
  arma::mat invsig=arma::inv(sig);
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::mat res(N,n);res.zeros();
  arma::rowvec rd(n),log_like(N),pro(n);
  
  res.row(0)=fstrow;
  log_like(0)=loglik(res.row(0),x,Y);
  arma::rowvec gx=dif(res.row(0),x,Y);
  double epp=std::pow(ep,0.5);
  for(int i=1;i<N;i++){
    rd=rd.randn()*C;
    pro=res.row(i-1)+0.5*ep*gx*sig+epp*rd;
    arma::rowvec gy=dif(pro,x,Y);
    arma::rowvec gp=(res.row(i-1)-pro-0.5*ep*gy*sig)/epp;
    double a=loglik(pro,x,Y);
    double acc=a-0.5*arma::sum(gp*invsig*gp.t())-(log_like(i-1)-0.5*arma::sum(rd*invsig*rd.t()));
    double u=std::log(arma::as_scalar(arma::randu(1)));
    if(acc>u){
      res.row(i)=pro;
      log_like(i)=a;
      gx=gy;
    }else{
      res.row(i)=res.row(i-1);
      log_like(i)=log_like(i-1);
    }
  }
  out(0)=log_like;out(1)=res;
  return(out);
}


// [[Rcpp::export]]
List fmala(arma::mat x,arma::rowvec Y,int N,double ep,arma::mat sigk,arma::rowvec vvk){
  List out(5),rm(2);
  int n=x.n_cols;
  arma::mat sig(n,n);sig=sigk;
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma:: mat  chsig=arma::chol(sig);
  arma::rowvec rd(n),lk(N),pro(n),pro_t(n),v(n),vv(n);vv=vvk;
  arma::mat res(N,n);res.row(0)=rd.randn();
  lk(0)=loglik(res.row(0),x,Y);
  int M=1;
  
  clock_t start=std::clock();
  rm=mala(x,Y,(N-M),sig,ep,res.row(0));
  arma::mat bm2=rm(1); arma::rowvec lk2=rm(0);
  res.rows(M,N-1)=bm2;lk.subvec(M,N-1)=lk2;
  int mm=bm2.n_cols,mn=bm2.n_rows;
  
  
  //res=bm2;lk=lk2;
  uvec IDX = regspace<uvec>(0,30,n-1);
  double t4 = (std::clock() - start )*1000/CLOCKS_PER_SEC;
  double acc=accept(bm2);
  Rcout << "The accept rate" << acc << "\n";
  arma::mat ress=res.cols(IDX);
  out(0)=lk;out(1)=ress;out(2)=t4/1000.0;//out(3)=sec(res.rows(1e4,N-1));
  return(out);
}

//pcn kernel 
// [[Rcpp::export]]
List pcn(arma::mat x,arma::rowvec Y,int N,arma::mat sig,arma::rowvec vv,double rho,
         arma::rowvec firstrow){
  List out(2);int n=x.n_cols;
  arma::mat C(n,n);C=arma::chol(sig);
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::mat invsig=arma::inv(sig);
  arma::mat res(N,n);res.row(0)=firstrow;
  arma::rowvec rd(n),log_like(N),pro(n),pro_t(n),v(N);
  log_like(0)=loglik(res.row(0),x,Y);
  double bb,bbn,r1,r2;
  
  r1=std::pow(rho,0.5);r2=std::pow(1.0-rho,0.5);
  arma:: rowvec orgv=(res.row(0)-vv)*invsnew;
  double aa=arma::sum(orgv%orgv);
  double gg=0;v(0)=1;
  for(int i=1;i<N;i++){
    rd=rd.randn()*C;
    pro=vv+r1*(res.row(i-1)-vv)+r2*rd;
    double a=loglik(pro,x,Y);
    double bb=arma::sum((pro-vv)*invsig*(pro-vv).t());
    double acc=a-log_like(i-1)+0.5*bb-0.5*aa;
    double u=std::log(arma::as_scalar(arma::randu(1)));
    if(acc>u){
      res.row(i)=pro;
      log_like(i)=a;
      aa=bb;
    }else{
      res.row(i)=res.row(i-1);
      log_like(i)=log_like(i-1);
    }
    //Rcout << "The accept rate" << i << "\n";
  }
  out(0)=log_like;out(1)=res;
  return(out);
}

// [[Rcpp::export]]
List fpcn(arma::mat x,arma::rowvec Y,int N,double rho,arma::mat sigk,arma::rowvec vvk){
  List out(5),rm(2);
  int n=x.n_cols;
  arma::mat sig(n,n);sig=sigk;
  
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::rowvec rd(n),lk(N),pro(n),pro_t(n),v(n),vv(n);vv=vvk;
  arma::mat res(N,n);res.row(0)=rd.randn();
  lk(0)=loglik(res.row(0),x,Y);
  int M=1;
  clock_t start=std::clock();
  
  rm=pcn(x,Y,(N-M),sig,vv,rho,res.row(0));
  arma::mat bm2=rm(1); arma::rowvec lk2=rm(0);
  res.rows(M,N-1)=bm2;lk.subvec(M,N-1)=lk2;
  //res=bm2;lk=lk2;
  uvec IDX = regspace<uvec>(0,30,n-1);
  double t4 = (std::clock() - start )*1000/CLOCKS_PER_SEC;
  double acc=accept(bm2);
  Rcout << "The accept rate" << acc << "\n";
  
  arma::mat ress=res.cols(IDX);
  out(0)=lk;out(1)=ress;out(2)=t4/1000.0;//out(3)=sec(res.rows(1e4,N-1));
  return(out);
}



/*pcn_mala kernel*/
// [[Rcpp::export]]
List pcnmala(arma::mat x,arma::rowvec Y,int N,double delta,
             arma::mat sig,arma::rowvec vv,arma::rowvec firstrow){
  List out(2);
  int n=x.n_cols;
  arma::mat clp=arma::chol(sig);
  arma::mat prim(n,n);prim=sig;
  arma::mat invsig=arma::inv(sig);
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  
  arma::mat res(N,n);res.zeros();
  res.row(0)=firstrow;
  arma::rowvec rd(n),log_like(N),pro(n),log_likef(N);
  
  arma::mat copm=res;
  copm.row(0)=res.row(0)-vv;
  arma::vec zl=loglikm(copm.row(0),x,Y,vv,invsig);
  log_like(0)=zl(1);
  log_likef(0)=zl(0);
  
  arma::rowvec gx(n),gy(n),zp(n),zpro(n);
  gx=dif(copm.row(0)+vv,x,Y)+(copm.row(0))*invsig;
  double r1=(2.0-delta)/(2.0+delta),r2=(2.0*delta)/(2.0+delta),r3=std::pow(8.0*delta,0.5)/(2.0+delta);
  
  for(int i=1;i<N;i++){
    rd=rd.randn()*clp;
    pro=r1*(copm.row(i-1))+r2*gx*prim+r3*rd;
    arma::vec aa=loglikm(pro,x,Y,vv,invsig);
    gy=dif(pro+vv,x,Y)+(pro)*invsig;
    double a=aa(0);
    double acc=a+0.5*arma::sum((copm.row(i-1)-pro)%gy)+delta/4.0*arma::sum((copm.row(i-1)+pro)%gy)-
      delta/4.0*arma::sum(gy*sig*gy.t())
      -log_likef(i-1)-(0.5*arma::sum((pro-copm.row(i-1))%gx)+delta/4.0*arma::sum((copm.row(i-1)+pro)%gx)-
        delta/4.0*arma::sum(gx*sig*gx.t()));
    double u=std::log(arma::as_scalar(arma::randu(1)));
    if(acc>u){
      copm.row(i)=pro;
      log_likef(i)=a;
      log_like(i)=aa(1);
      res.row(i)=pro+vv;
      gx=gy;
    }else{
      copm.row(i)=copm.row(i-1);
      log_likef(i)=log_likef(i-1);
      log_like(i)=log_like(i-1);
      res.row(i)=res.row(i-1);
    }
  }
  out(0)=log_like;out(1)=res;
  return(out);
}

// [[Rcpp::export]]
List fpcnmala(arma::mat x,arma::rowvec Y,int N,double ep,arma::mat sigk,arma::rowvec vvk){
  List out(5),rm(2);
  int n=x.n_cols;
  arma::mat sig(n,n);sig=sigk;
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::rowvec rd(n),lk(N),pro(n),pro_t(n),v(n),vv(n);vv=vvk;
  arma::mat res(N,n);res.row(0)=rd.randn();
  lk(0)=loglik(res.row(0),x,Y);
  int M=1;
  rm=pcnmala(x,Y,M,ep,sig,vv,res.row(0));
  Rcout << "The ep" << ep << "\n";
  
  clock_t start=std::clock();
  rm=pcnmala(x,Y,(N-M),ep,sig,vv,res.row(0));
  arma::mat bm2=rm(1); arma::rowvec lk2=rm(0);
  res.rows(M,N-1)=bm2;lk.subvec(M,N-1)=lk2;
  //res=bm2;lk=lk2;
  uvec IDX = regspace<uvec>(0,30,n-1);
  double t4 = (std::clock() - start )*1000/CLOCKS_PER_SEC;
  double acc=accept(bm2);
  
  //int mm=bm.n_cols,mn=bm.n_rows;
  Rcout << "The accept rate" << acc << "\n";
  arma::mat ress=res.cols(IDX);
  out(0)=lk;out(1)=ress;out(2)=t4/1000.0;//out(3)=sec(res.rows(1e4,N-1));
  return(out);
}


/* pcn-margin kernel*/
// [[Rcpp::export]]
List pcnmargin(arma::mat x,arma::rowvec Y,int N,double delta,
               arma::mat sig,arma::rowvec vv,arma::rowvec firstrow){
  int n=x.n_cols;
  arma::mat ssig=sig;
  List out(2);
  
  arma::mat C(n,n);C=arma::chol(sig);
  arma::mat invsig=arma::inv(sig);
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::mat res(N,n);res.zeros();
  res.row(0)=firstrow;
  arma::rowvec rd(n),log_like(N),pro(n),log_likef(N);
  
  
  arma::mat I(n,n);
  I=I.eye();
  arma::mat A=delta/2.0*arma::inv_sympd(sig+0.5*delta*I)*sig;
  arma::mat pm=(2.0/delta*A*A.t()+A);
  arma::mat iva=arma::inv(2.0/delta*A+I);
  arma::mat cha=arma::chol(pm);
  
  arma::mat copm=res;
  copm.row(0)=res.row(0)-vv;
  arma::vec zl=loglikm(copm.row(0),x,Y,vv,invsig);
  log_like(0)=zl(1);
  log_likef(0)=zl(0);
  arma::rowvec gx(n),gy(n),zp(n),zpro(n);
  gx=dif(copm.row(0)+vv,x,Y)+(copm.row(0))*invsig;
  for(int i=1;i<N;i++){
    zpro=copm.row(i-1)+0.5*delta*gx;
    pro=2.0/delta*(zpro)*A+rd.randn()*cha;
    arma::vec aa=loglikm(pro,x,Y,vv,invsig);
    gy=dif(pro+vv,x,Y)+(pro)*invsig;
    double a=aa(0);
    double acc=a+arma::sum((copm.row(i-1)-2.0/delta*(pro+delta/4.0*gy)*A)*iva*gy.t())-
      log_likef(i-1)-arma::sum((pro-2.0/delta*(copm.row(i-1)+delta/4.0*gx)*A)*iva*gx.t());
    double u=std::log(arma::as_scalar(arma::randu(1)));
    if(acc>u){
      copm.row(i)=pro;
      log_likef(i)=a;
      log_like(i)=aa(1);
      gx=gy;
      res.row(i)=vv+pro;
    }else{
      copm.row(i)=copm.row(i-1);
      log_likef(i)=log_likef(i-1);
      log_like(i)=log_like(i-1);
      res.row(i)=res.row(i-1);
    }
  }
  out(0)=log_like;out(1)=res;
  return(out);
}

// [[Rcpp::export]]
List fpcnmargin(arma::mat x,arma::rowvec Y,int N,double ep,arma::mat sigk,arma::rowvec vvk){
  List out(5),rm(2);
  int n=x.n_cols;
  arma::mat sig(n,n);sig=sigk;
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::rowvec rd(n),lk(N),pro(n),pro_t(n),v(n),vv(n);vv=vvk;
  arma::mat res(N,n);res.row(0)=rd.randn();
  lk(0)=loglik(res.row(0),x,Y);
  int M=1;
  Rcout << "The ep" << ep << "\n";
  
  clock_t start=std::clock();
  rm=pcnmargin(x,Y,(N-M),ep,sig,vv,res.row(0));
  arma::mat bm2=rm(1); arma::rowvec lk2=rm(0);
  res.rows(M,N-1)=bm2;lk.subvec(M,N-1)=lk2;
  //res=bm2;lk=lk2;
  uvec IDX = regspace<uvec>(0,30,n-1);
  double t4 = (std::clock() - start )*1000/CLOCKS_PER_SEC;
  double acc=accept(bm2);
  Rcout << "The accept rate" << acc << "\n";
  arma::mat ress=res.cols(IDX);
  out(0)=lk;out(1)=ress;out(2)=t4/1000.0;//out(3)=sec(res.rows(1e4,N-1));
  return(out);
}


// split hmc kernel
// [[Rcpp::export]]
List split_hmc(arma::mat x,arma::rowvec Y,int N,double ep,int L,arma::mat sigk,arma::rowvec vvk,
               arma::rowvec firstrow){
  
  int n=x.n_cols;
  arma::mat sig=sigk;
  List out(2);
  
  arma::mat clp(n,n);clp=arma::chol(sig);
  arma::mat invsig=arma::inv(sig);
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::mat res(N,n);res.zeros();
  res.row(0)=firstrow;
  arma::rowvec rd(n),log_like(N),pro(n),vv(n);vv=vvk;
  arma::rowvec zp(n),zpro(n),zm_new(n),zv_new(n);
  log_like(0)=loglik(res.row(0),x,Y);
  arma::mat zm(L+1,n),zv(L+1,n);
  arma::mat comp=res;
  comp.row(0)=res.row(0)-vv;
  arma::rowvec diff;
  for(int i=1;i<N;i++){
    zm.zeros();
    zv.zeros();
    zv.row(0)=rd.randn()*clp;
    zm.row(0)=comp.row(i-1);
    //zv.row(1)=zv.row(0)-0.5*ep*dif(zm.row(0));
    for(int j=1;j<(L+1);j++){
      
      diff=dif(zm.row(j-1)+vv,x,Y)+(zm.row(j-1))*invsig;
      zv.row(j-1)=zv.row(j-1)+0.5*ep*diff*sig;
      
      zm_new=std::cos(ep)*zm.row(j-1)+std::sin(ep)*zv.row(j-1);
      zv_new=-std::sin(ep)*zm.row(j-1)+std::cos(ep)*zv.row(j-1);
      
      diff=dif(zm_new+vv,x,Y)+(zm_new)*invsig;
      zv_new=zv_new+0.5*ep*diff*sig;
      
      zm.row(j)=zm_new;
      zv.row(j)=zv_new;
      
      
    }
    
    pro=zm.row(L);
    double a=loglik(pro+vv,x,Y);
    double acc=a-log_like(i-1)-0.5*arma::as_scalar((zv.row(L))*invsig*(zv.row(L)).t())+
      0.5*arma::as_scalar((zv.row(0))*invsig*(zv.row(0)).t());
    double u=std::log(arma::as_scalar(arma::randu(1)));
    if(acc>u){
      res.row(i)=pro+vv;
      log_like(i)=a;
      comp.row(i)=pro;
    }else{
      res.row(i)=res.row(i-1);
      log_like(i)=log_like(i-1);
      comp.row(i)=comp.row(i-1);
    }
  }
  out(0)=log_like;out(1)=res;
  return(out);
}

// [[Rcpp::export]]
List fsplit_hmc(arma::mat x,arma::rowvec Y,int N,double ep,int L,arma::mat sigk,arma::rowvec vvk){
  List out(5),rm(2);
  int n=x.n_cols;
  arma::mat sig(n,n);sig=sigk;
  arma::mat chl=arma::chol(sig);
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::rowvec rd(n),lk(N),pro(n),pro_t(n),v(n),vv(n);vv=vvk;
  arma::mat res(N,n);res.row(0)=rd.randn();
  lk(0)=loglik(res.row(0),x,Y);
  int M=1;
  
  Rcout << "The ep" << ep << "\n";
  
  clock_t start=std::clock();
  
  rm=split_hmc(x,Y,(N-M),ep,L,sig,vv,res.row(0));
  arma::mat bm2=rm(1); arma::rowvec lk2=rm(0);
  res.rows(M,N-1)=bm2;lk.subvec(M,N-1)=lk2;

  uvec IDX = regspace<uvec>(0,30,n-1);
  double t4 = (std::clock() - start )*1000/CLOCKS_PER_SEC;
  double acc=accept(bm2);
  Rcout << "The accept rate" << acc << "\n";
  arma::mat ress=res.cols(IDX);
  out(0)=lk;out(1)=ress;out(2)=t4/1000.0;//out(3)=sec(res.rows(1e4,N-1));
  return(out);
}



