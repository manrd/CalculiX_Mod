/*     CalculiX - A 3-dimensional finite element program                 */
/*              Copyright (C) 1998-2015 Guido Dhondt                          */

/*     This program is free software; you can redistribute it and/or     */
/*     modify it under the terms of the GNU General Public License as    */
/*     published by the Free Software Foundation(version 2);    */
/*                    */

/*     This program is distributed in the hope that it will be useful,   */
/*     but WITHOUT ANY WARRANTY; without even the implied warranty of    */ 
/*     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the      */
/*     GNU General Public License for more details.                      */

/*     You should have received a copy of the GNU General Public License */
/*     along with this program; if not, write to the Free Software       */
/*     Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.         */

#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <pthread.h>
#include <omp.h>
#include "CalculiX.h"

void resultsini(ITG *nk,double *v,ITG *ithermal,char *filab,ITG *iperturb,
		double *f,double *fn,ITG *nactdof,ITG *iout,double *qa,
		double *vold,double *b,ITG *nodeboun,ITG *ndirboun,
		double *xboun,ITG *nboun,ITG *ipompc,ITG *nodempc,
		double *coefmpc,char *labmpc,ITG *nmpc,ITG *nmethod,
		double *cam,ITG *neq,double *veold,double *accold,
		double *bet,double *gam,double *dtime,ITG *mi,double *vini,
		ITG *nprint,char *prlab,ITG *intpointvarm,ITG *calcul_fn,
		ITG *calcul_f,ITG *calcul_qa,ITG *calcul_cauchy,ITG *ikin,
		ITG *intpointvart,char *typeboun,ITG *num_cpus,ITG *mortar,
		ITG *nener,ITG *iponoeln,ITG *network){

  ITG mt,i,j,node,ndir,ist,index,incrementalmpc;

  double bnac,fixed_disp;
    
  mt=mi[1]+1;

  if((*iout!=2)&&(*iout>-1)){
    if((*nmethod!=4)||(iperturb[0]<=1)){

      /* steady state: mechanical (static).
         v-accumulation is per-node disjoint; cam[0]/cam[3] keep the
         first strict maximum in scan order (ordered thread merge
         reproduces the serial result bit-identically). */

      if(ithermal[0]!=2){
	ITG _nt=*num_cpus; ITG _t;
	double _lmax[64]; double _lidx[64]; int _lhit[64];
	int _track=(iperturb[0]!=0)&&(abs(*nmethod)==1);
	if(_nt<1) _nt=1;
	if(_nt>64) _nt=64;
	#pragma omp parallel num_threads(_nt)
	{
	  int _tid=omp_get_thread_num();
	  int _nth=omp_get_num_threads();
	  ITG _chunk=(*nk+_nth-1)/_nth;
	  ITG _s=_tid*_chunk,_e=_s+_chunk;
	  double _lv=-1.,_li=0.; int _hit=0;
	  if(_e>*nk) _e=*nk;
	  for(ITG _i=_s;_i<_e;_i++){
	    for(ITG _j=1;_j<mt;_j++){
	      ITG _d=nactdof[mt*_i+_j];
	      double _b;
	      if(_d<=0) continue;
	      _b=b[_d-1];
	      v[mt*_i+_j]+=_b;
	      if(_track){
		double _a=fabs(_b);
		if(_a>_lv){_lv=_a;_li=_d-0.5;_hit=1;}
	      }
	    }
	  }
	  _lmax[_tid]=_lv; _lidx[_tid]=_li; _lhit[_tid]=_hit;
	}
	if(_track){
	  for(_t=0;_t<_nt;_t++){
	    if(_lhit[_t]&&_lmax[_t]>cam[0]){cam[0]=_lmax[_t];cam[3]=_lidx[_t];}
	  }
	}
      }

      /* steady state: thermal (same disjointness + ordered-merge) */

      if(ithermal[0]>1){
	ITG _nt=*num_cpus; ITG _t;
	double _lmax[64]; double _lidx[64]; int _lhit[64];
	int _track=(iperturb[0]!=0)&&(abs(*nmethod)==1);
	if(_nt<1) _nt=1;
	if(_nt>64) _nt=64;
	#pragma omp parallel num_threads(_nt)
	{
	  int _tid=omp_get_thread_num();
	  int _nth=omp_get_num_threads();
	  ITG _chunk=(*nk+_nth-1)/_nth;
	  ITG _s=_tid*_chunk,_e=_s+_chunk;
	  double _lv=-1.,_li=0.; int _hit=0;
	  if(_e>*nk) _e=*nk;
	  for(ITG _i=_s;_i<_e;_i++){
	    ITG _d;
	    double _b;
	    if(nactdof[mt*_i]<=0) continue;
	    _d=nactdof[mt*_i];
	    _b=b[_d-1];
	    v[mt*_i]+=_b;
	    if(_track){
	      double _a=fabs(_b);
	      if(_a>_lv){_lv=_a;_li=_d-0.5;_hit=1;}
	    }
	  }
	  _lmax[_tid]=_lv; _lidx[_tid]=_li; _lhit[_tid]=_hit;
	}
	if(_track){
	  for(_t=0;_t<_nt;_t++){
	    if(_lhit[_t]&&_lmax[_t]>cam[1]){cam[1]=_lmax[_t];cam[4]=_lidx[_t];}
	  }
	}
      }
    }else{

      /* direct integration dynamic step
	 b contains the acceleration increment */

      if(ithermal[0]!=2){
	iniparll(&mt,nactdof,b,v,veold,accold,bet,gam,
		 dtime,cam,nk,num_cpus,mortar);
      }
	    
      /* transient thermal step */

      if(ithermal[0]>1){
	/* cam[1]/cam[4] unconditionally tracked here (unlike static
	   above); cam[2] is a plain max. Ordered merge reproduces serial. */
	ITG _nt=*num_cpus; ITG _t;
	double _lmax1[64]; double _lidx1[64]; int _lhit1[64];
	double _lmax2[64]; int _lhit2[64];
	if(_nt<1) _nt=1;
	if(_nt>64) _nt=64;
	#pragma omp parallel num_threads(_nt)
	{
	  int _tid=omp_get_thread_num();
	  int _nth=omp_get_num_threads();
	  ITG _chunk=(*nk+_nth-1)/_nth;
	  ITG _s=_tid*_chunk,_e=_s+_chunk;
	  double _lv1=-1.,_li1=0.,_lv2=-1.; int _hit1=0,_hit2=0;
	  if(_e>*nk) _e=*nk;
	  for(ITG _i=_s;_i<_e;_i++){
	    ITG _d;
	    double _b,_dv;
	    veold[mt*_i]=0.;
	    if(nactdof[mt*_i]<=0) continue;
	    _d=nactdof[mt*_i];
	    _b=b[_d-1];
	    v[mt*_i]+=_b;
	    if(fabs(_b)>_lv1){_lv1=fabs(_b);_li1=_d-0.5;_hit1=1;}
	    /* maximum change is not taken into account for network nodes */
	    if((*network<=0)||(iponoeln[_i]!=0)){
	      _dv=fabs(v[mt*_i]-vini[mt*_i]);
	      if(_dv>_lv2){_lv2=_dv;_hit2=1;}
	    }
	  }
	  _lmax1[_tid]=_lv1; _lidx1[_tid]=_li1; _lhit1[_tid]=_hit1;
	  _lmax2[_tid]=_lv2; _lhit2[_tid]=_hit2;
	}
	for(_t=0;_t<_nt;_t++){
	  if(_lhit1[_t]&&_lmax1[_t]>cam[1]){cam[1]=_lmax1[_t];cam[4]=_lidx1[_t];}
	  if(_lhit2[_t]&&_lmax2[_t]>cam[2])cam[2]=_lmax2[_t];
	}
      }
    }
  }

  /* initialization */

  *calcul_fn=0;
  *calcul_f=0;
  *calcul_qa=0;
  *calcul_cauchy=0;

  /* procedure requirements */
    
  if((iperturb[0]>=2)||((iperturb[0]<=0)&&(*iout<0))){
    if((*iout<1)&&(*iout>-2)){
      *calcul_fn=1;
      *calcul_f=1;
      *calcul_qa=1;
    }else if((*iout!=-2)&&(iperturb[1]==1)){
      *calcul_cauchy=1;
    }
  }

  /* output requests */
    
  if(*iout>0){
    if((strcmp1(&filab[348],"RF  ")==0)||(strcmp1(&filab[783],"RFL ")==0)||
       (strcmp1(&filab[2610],"PRF ")==0)){
      *calcul_fn=1;
    }
    if(*calcul_fn!=1){
      for(i=0;i<*nprint;i++){
	if((strcmp1(&prlab[6*i],"RF  ")==0)||
	   (strcmp1(&prlab[6*i],"RFL ")==0)||
	   (strcmp1(&prlab[6*i],"RECU")==0)){
	  *calcul_fn=1;
	  break;
	}
      }
    }
  }

  /* initializing fn */

  if(*calcul_fn==1){
    setpardou(fn,0.,mt**nk,*num_cpus);
  }
    
  /* initializing f */

  if(*calcul_f==1){
    setpardou(f,0.,*neq,*num_cpus);
  }

  /* SPC's and MPC's have to be taken into account for 
     iout=0,1 and -1 */

  if(abs(*iout)<2){

    /* single point constraints */

    for(i=0;i<*nboun;i++){
      if((ndirboun[i]>mi[1])||(typeboun[i]=='F')) continue;
      fixed_disp=xboun[i];
	    
      /* a discontinuity in the displacements in an initial
	 acceleration step (recognized by the "special" time
	 increment) should not lead to a change in 
	 the acceleration or velocity; actually, such a 
	 discontinuity is not allowed since it leads to
	 infinite accelerations */
					      
      if((*nmethod==4)&&(iperturb[0]>1)){
	ndir=ndirboun[i];
	node=nodeboun[i]-1;
	if(ndir>0){
	  if(*mortar!=-1){
	    
	    /* bnac is the change in acceleration */
		    
	    bnac=(xboun[i]-v[mt*node+ndir])/(*bet**dtime**dtime);
	    if((ITG)floor(*dtime*1.e28+0.5)!=123571113){
	      veold[mt*node+ndir]+=*gam**dtime*bnac;
	      accold[mt*node+ndir]+=bnac;
	    }
	  }else{
	    
	    /* massless contact: bnac is the velocity */

	    bnac=(xboun[i]-v[mt*node+ndir])/(*dtime);
	    veold[mt*node+ndir]=bnac;
	  }
	}
      }
      v[mt*(nodeboun[i]-1)+ndirboun[i]]=fixed_disp;
    }

    /* multiple point constraints */

    /* the parameter incrementalmpc indicates whether the
       incremental displacements enter the mpc or the total 
       displacements (incrementalmpc=0) */
	
    for(i=0;i<*nmpc;i++){

      if((strcmp1(&labmpc[20*i],"                    ")==0)||
	 (strcmp1(&labmpc[20*i],"CYCLIC")==0)||
	 (strcmp1(&labmpc[20*i],"SUBCYCLIC")==0)){
	incrementalmpc=0;
      }else{
	if((*nmethod==2)||(*nmethod==3)||
	   ((iperturb[0]==0)&&(abs(*nmethod)==1))){
	  incrementalmpc=0;
	}else{
	  incrementalmpc=1;
	}
      }

      ist=ipompc[i]-1;
      node=nodempc[3*ist]-1;
      ndir=nodempc[3*ist+1];
	    
      if(ndir==0){
	if(ithermal[0]<2) continue;
      }else if(ndir>mi[1]){
	continue;
      }else{
	if(ithermal[0]==2) continue;
      }

      index=nodempc[3*ist+2]-1;
      fixed_disp=0.;
      if(index!=-1){
	do{
	  if(incrementalmpc==0){
	    fixed_disp-=coefmpc[index]*
	      v[mt*(nodempc[3*index]-1)+nodempc[3*index+1]];
	  }else{
	    fixed_disp-=coefmpc[index]*
	      (v[mt*(nodempc[3*index]-1)+nodempc[3*index+1]]-
	       vold[mt*(nodempc[3*index]-1)+nodempc[3*index+1]]);
	  }
	  index=nodempc[3*index+2]-1;
	  if(index==-1) break;
	}while(1);
      }
      fixed_disp/=coefmpc[ist];
      if(incrementalmpc==1){
	fixed_disp+=vold[mt*node+ndir];
      }

      /* a discontinuity in the displacements in an initial
	 acceleration step (recognized by the "special" time
	 increment) should not lead to a change in 
	 the acceleration or velocity; actually, such a 
	 discontinuity is not allowed since it leads to
	 infinite accelerations */

      if((*nmethod==4)&&(iperturb[0]>1)){
	if(ndir>0){
	  if(*mortar!=-1){
	    
	    /* bnac is the change in acceleration */

	    bnac=(fixed_disp-v[mt*node+ndir])/(*bet**dtime**dtime);
	    if((ITG)floor(*dtime*1.e28+0.5)!=123571113){
	      veold[mt*node+ndir]+=*gam**dtime*bnac;
	    }

	    /* different treatment than for xboun: accold is
	       not included in the above if statement */
		    
	    accold[mt*node+ndir]+=bnac;
	  }else{
	    
	    /* massless contact: bnac is the velocity */

	    bnac=(fixed_disp-v[mt*node+ndir])/(*dtime);
	    veold[mt*node+ndir]=bnac;
	  }
	}
      }
      v[mt*node+ndir]=fixed_disp;
    }
  }

  /* check whether the kinetic energy has to be calculated  */

  *ikin=0;

  /* dynamic calculations */
    
  if((*nmethod==4)&&(iperturb[0]>1)&&(ithermal[0]<=1)&&(*nener==1)){
    *ikin=1;
  }
    
  /* output requests */

  if(*ikin!=1){
    for(i=0;i<*nprint;i++){
      if(strcmp1(&prlab[6*i],"ELKE")==0){
	*ikin=1;
      }
    }
  }

  qa[0]=0.;
  qa[1]=0.;

  *intpointvarm=1;
  *intpointvart=1;

  if((*nmethod>=4)&&(*nmethod<=5)&&(iperturb[0]<2)){
    *intpointvarm=0;
    if((strcmp1(&filab[174],"S")==0)||
       (strcmp1(&filab[261],"E")==0)||
       (strcmp1(&filab[348],"RF")==0)||
       (strcmp1(&filab[435],"PEEQ")==0)||
       (strcmp1(&filab[522],"ENER")==0)||
       (strcmp1(&filab[609],"SDV")==0)||
       (strcmp1(&filab[1044],"ZZS")==0)||
       (strcmp1(&filab[1044],"ERR")==0)||
       (strcmp1(&filab[1479],"PHS")==0)||
       (strcmp1(&filab[1653],"MAXS")==0)||
       (strcmp1(&filab[2175],"CONT")==0)||
       (strcmp1(&filab[2262],"CELS")==0)||
       (strcmp1(&filab[2610],"PRF")==0)){
      *intpointvarm=1;
    }
    if(*intpointvarm!=1){
      for(i=0;i<*nprint;i++){
	if((strcmp1(&prlab[6*i],"S")==0)||
	   (strcmp1(&prlab[6*i],"E")==0)||
	   (strcmp1(&prlab[6*i],"PEEQ")==0)||
	   (strcmp1(&prlab[6*i],"ENER")==0)||
	   (strcmp1(&prlab[6*i],"ELKE")==0)||
	   (strcmp1(&prlab[6*i],"CDIS")==0)||
	   (strcmp1(&prlab[6*i],"CSTR")==0)||
	   (strcmp1(&prlab[6*i],"CELS")==0)||
	   (strcmp1(&prlab[6*i],"SDV")==0)||
	   (strcmp1(&prlab[6*i],"RF")==0)){
	  *intpointvarm=1;
	  break;
	}
      }
    }

    *intpointvart=0;
    if((strcmp1(&filab[696],"HFL")==0)||
       (strcmp1(&filab[783],"RFL")==0)){
      *intpointvart=1;
    }
    if(*intpointvart!=1){
      for(i=0;i<*nprint;i++){
	if((strcmp1(&prlab[6*i],"HFL")==0)||
	   (strcmp1(&prlab[6*i],"RFL")==0)||
	   (strcmp1(&prlab[6*i],"RECU")==0)){
	  *intpointvart=1;
	  break;
	}
      }
    }

    /* if internal forces are requested integration point
       values have to be calculated */

    if(*calcul_fn==1){
      *intpointvarm=1;
      *intpointvart=1;
    }
  }
    
  return;
}
