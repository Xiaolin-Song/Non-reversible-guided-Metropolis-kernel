#include <RcppArmadillo.h>
#include <Rcpp/Benchmark/Timer.h>
// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadilloExtensions/sample.h>
using namespace Rcpp;
using namespace arma;

static double const log2pi = std::log(2.0 * M_PI);


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


//loglikelihood


// [[Rcpp::export]]
double loglik(arma::rowvec x,arma::mat yy,double lambda){
  int nrow=yy.n_rows,ncol=yy.n_cols;
  int n=x.size();
  int k=n-1;
  double lik=0,alpha=1.0,beta=0.5;//alpha=x(k);
  Rcpp::NumericVector y=Rcpp::as<Rcpp::NumericVector>(wrap(yy.row(0)));
  for(int i=0;i<nrow;i++){
    y=Rcpp::as<Rcpp::NumericVector>(wrap(yy.row(i)));
    y=factorial(y);
    for(int j=0;j<ncol;j++){
      lik=lik+yy(i,j)*std::log(x(i))-x(i)-std::log(y(j))-alpha*std::log(beta)+
        +(alpha-1.0)*log(x(i))-x(i)/beta-alpha*lambda-std::log(tgamma(alpha));
    }
  }
  return(lik);
}

// [[Rcpp::export]]
List fcsk(int N,arma::mat yy,double lambda,double rho){

  List out(4);
  auto start = std::chrono::steady_clock::now();
  int nrow=yy.n_rows,ncol=yy.n_cols;
  int n=nrow;
  //uvec IDX = regspace<uvec>(0,40,n-1);
  
  arma::mat res(N,n);
  arma::vec log_like(N);
  arma::rowvec rd(n),b(n),ypro(n);
  rd=rd.randu();//rd(n-1)=10*rd(n-1);
  res.row(0)=rd;
  log_like(0)=loglik(res.row(0),yy,lambda);
  double pn,pnn;
  arma::rowvec ls=arma::log(res.row(0));
  pn=0.5*arma::accu(res.row(0));
  
  for(int i=1;i<N;i++){
    
    b=b.randn();
    for(int kk=0;kk<n;kk++){
      ypro(kk)=rho*res(i-1,kk)+(1.0-rho)*b(kk)*b(kk)+2.0*std::sqrt((1.0-rho)*rho*res(i-1,kk))*b(kk);
    }
    
    double a=loglik(ypro,yy,lambda);
    arma::rowvec lp=arma::log(ypro);
    pnn=0.5*arma::accu(ypro);
    double acc=a-log_like(i-1)-(pn-pnn)-0.5*arma::accu(ls-lp);
    double u=std::log(arma::as_scalar(arma::randu(1)));
    if(acc>u){
      res.row(i)=ypro;
      pn=pnn;
      ls=lp;
      log_like(i)=a;
    }else{
      res.row(i)=res.row(i-1);
      log_like(i)=log_like(i-1);
    }
  }
  
  auto end = std::chrono::steady_clock::now();
  double t4=std::chrono::duration <double> (end - start).count();
  //arma::mat ress=res.cols(IDX);
  
  out(0)=log_like;out(1)=res;
  out(2)=t4;
  out(3)=rho;
  return(out);
}



// [[Rcpp::export]]
List fcsk_h(int N,arma::mat yy,double lambda,double rho){

  List out(4);
  auto start = std::chrono::steady_clock::now();
  int nrow=yy.n_rows,ncol=yy.n_cols;
  int n=nrow;
  
  arma::mat res(N,n);  
  arma::vec log_like(N);
  arma::rowvec rd(n),b(n),ypro(n),yk(n);
  double v=1.0;
  rd=rd.randu();//rd(n-1)=10*rd(n-1);
  
  res.row(0)=rd;
  log_like(0)=loglik(res.row(0),yy,lambda);
  
  double pn,pnn;
  
  double bb=arma::sum(res.row(0));
  arma::rowvec ls=arma::log(res.row(0)),lp(n);
  pn=0.5*arma::accu(ls);
  
  double g,bnew;
  for(int i=1;i<N;i++){
    
    
    b=b.randn();
    bnew=0;
    g=R::rgamma(0.5*n,2.0/bb);
    for(int kk=0;kk<n;kk++){
      ypro(kk)=rho*res(i-1,kk)+(1.0-rho)/g*b(kk)*b(kk)+2.0*std::sqrt((1.0-rho)*rho*res(i-1,kk)/g)*b(kk);
      bnew +=ypro(kk);
    }
    
    lp=arma::log(ypro);
    double a=loglik(ypro,yy,lambda);
    pnn=0.5*arma::accu(lp);
    double acc=a-log_like(i-1)+0.5*n*std::log(bnew/bb)+(pnn-pn);
    double u=std::log(arma::as_scalar(arma::randu(1)));
    
    
    if(acc>u){
      res.row(i)=ypro;
      log_like(i)=a;
      bb=bnew;
      pn=pnn;
      ls=lp;
    }else{
      log_like(i)=log_like(i-1);
      res.row(i)=res.row(i-1);
    }
    
  }
  auto end = std::chrono::steady_clock::now();
  double t4=std::chrono::duration <double> (end - start).count();
  
  out(0)=log_like;out(1)=res;
  out(2)=t4;
  out(3)=rho;
  return(out);
}

// [[Rcpp::export]]
List fcsk_g2(int N,arma::mat yy,double lambda,double rho){

  List out(5);
  auto start = std::chrono::steady_clock::now();
  int nrow=yy.n_rows,ncol=yy.n_cols;
  int n=nrow;
  arma::mat res(N,n);  
  arma::vec log_like(N);
  arma::rowvec rd(n),b(n),ypro(n),yk(n),vv(N);
  double v=1.0;
  rd=rd.randu();//rd(n-1)=10*rd(n-1);
  vv(0)=v;
  res.row(0)=rd;
  log_like(0)=loglik(rd,yy,lambda);
  
  double pn,pnn;
  
  double bb=arma::sum(res.row(0));
  arma::rowvec ls=arma::log(res.row(0)),lp(n);
  pn=0.5*arma::accu(ls);
  
  double g,bnew;
  for(int i=1;i<N;i++){
    
    
    b=b.randn();
    bnew=0;
    g=R::rgamma(0.5*n,2.0/bb);
    for(int kk=0;kk<n;kk++){
      ypro(kk)=rho*res(i-1,kk)+(1.0-rho)/g*b(kk)*b(kk)+2.0*std::sqrt((1.0-rho)*rho*res(i-1,kk)/g)*b(kk);
      bnew +=ypro(kk);
    }
    
    while((bnew-bb)*v>0){
      b=b.randn();
      g=R::rgamma(0.5*n,2.0/bb);
      bnew=0;
      for(int kk=0;kk<2;kk++){
        ypro(kk)=rho*res(i-1,kk)+(1.0-rho)/g*b(kk)*b(kk)+2.0*std::sqrt((1.0-rho)*rho/g*res(i-1,kk))*b(kk);
        bnew +=ypro(kk);
      }
    }
    
    lp=arma::log(ypro);
    double a=loglik(ypro,yy,lambda);
    pnn=0.5*arma::accu(lp);
    double acc=a-log_like(i-1)+0.5*n*std::log(bnew/bb)+(pnn-pn);
    double u=std::log(arma::as_scalar(arma::randu(1)));
    
    
    if(acc>u){
      res.row(i)=ypro;
      log_like(i)=a;
      bb=bnew;
      pn=pnn;
      ls=lp;
      vv(i)=vv(i-1);
    }else{
      log_like(i)=log_like(i-1);
      res.row(i)=res.row(i-1);
      v=-v;
      vv(i)=v;
    }
    
  }
  auto end = std::chrono::steady_clock::now();
  double t4=std::chrono::duration <double> (end - start).count();
  
  out(0)=log_like;out(1)=res;
  out(2)=t4;
  out(3)=rho;
  out(4)=vv;
  return(out);
}

// [[Rcpp::export]]
double js(arma::mat mx){
  int n=mx.n_rows;
  double js=0;
  arma::rowvec zd=mx.row(0);
  double ms;
  for(int i=1;i<n;i++){
    zd=mx.row(i)-mx.row(i-1);
    js=arma::sum(zd%zd)+js;
    // if((i%1000)==0){
    //   Rcout << "The value of v : " << i << "\n";
    // }
  }
  return(js/(1.0*n));
}

