/*     CalculiX - A 3-dimensional finite element program                 */
/*              Copyright (C) 1998-2015 Guido Dhondt                          */

/*     This program is free software; you can redistribute it and/or     */
/*     modify it under the terms of the GNU General Public License as    */
/*     published by the Free Software Foundation(version 2);    */
/*                                                                       */

/*     This program is distributed in the hope that it will be useful,   */
/*     but WITHOUT ANY WARRANTY; without even the implied warranty of    */
/*     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the      */
/*     GNU General Public License for more details.                      */

/*     You should have received a copy of the GNU General Public License */
/*     along with this program; if not, write to the Free Software       */
/*     Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.         */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "CalculiX.h"

/* Cost model: upper-triangle pair count of the element stiffness,
   N=(ndof*nope)*(ndof*nope+1)/2. This is exactly the number of
   add_sm_st scatter calls per clean element per Newton iteration
   (mafillsm.f) and tracks e_c3d/resultsmech work monotonically
   (bigger N <-> more nodes x more Gauss points). */
static long long elem_cost(const char *l8){

  int nope=8,ndof=3;

  if(l8[0]=='E'){
    /* contact springs: cost varies wildly. Face-to-face elements
       ('C' at 7th label position) run full segment integration in
       e_c3d/springstiff (mafillsm.f) with kon(indexe) nodes for
       mortar==1: C3D8-class cost. Node-to-face springs use the small
       digit+1 node count. Advection ('A') is skipped by workers but
       kept nonzero for partition sanity. */
    if(l8[6]=='C') return 300;
    {
      int d=(unsigned char)l8[7]-'0'+1;
      long long n;
      if(d<1) d=1;
      if(d>30) d=30;
      n=(long long)3*d;
      return n*(n+1)/2;
    }
  }
  if(l8[0]=='F'){
    /* fluid network elements skipped by workers; keep small */
    return 8;
  }
  if(l8[0]=='U'){
    /* user element: ndof = 7th label entry, nodes = 8th (see mastruct.c) */
    ndof=(unsigned char)l8[6];
    nope=(unsigned char)l8[7];
    if(nope<1) nope=1;
    if(ndof<1) ndof=1;
  }else if(l8[3]=='8'&&l8[4]=='I'){nope=11;}
  else if(l8[3]=='2'&&l8[4]=='0'){nope=20;}
  else if(l8[3]=='8'){nope=8;}
  else if(l8[3]=='1'&&l8[4]=='0'){nope=10;}
  else if(l8[3]=='4'||l8[2]=='4'){nope=4;}
  else if(l8[3]=='1'&&l8[4]=='5'){nope=15;}
  else if(l8[3]=='6'){nope=6;}
  else{
    /* beams/shells (expanded) or unknown: C3D8-equivalent */
    return 300;
  }
  {
    long long n=(long long)ndof*nope;
    return n*(n+1)/2;
  }
}

void elementcpuload(ITG *neapar,ITG *nebpar,ITG *ne,ITG *ipkon,char *lakon,
                    ITG *num_cpus){

 /*  divides the elements into contiguous ranges of approximately
     equal computational cost (sum of elem_cost over active elements)
     for parallel processing on different cpus. Element numbering may
     have gaps (ipkon[i]<0 skipped). Contiguity preserves mesh locality.
     lakon==NULL falls back to the legacy equal-element-count split. */

    ITG i,nepar,*ipar=NULL,idelta,isum;
    ITG t,nthr;
    long long *wpar=NULL,total,cum,target;

    /* CCX_EQUAL_PART=1 forces the legacy equal-element-count split
       (diagnostic: A/B partition comparison within one binary) */
    if(lakon!=NULL){
      static int eqpart_init=0,eqpart=0;
      if(!eqpart_init){
        const char *e=getenv("CCX_EQUAL_PART");
        eqpart=(e&&e[0]!='\0'&&e[0]!='0')?1:0;
        eqpart_init=1;
      }
      if(eqpart) lakon=NULL;
    }
    
    NNEW(ipar,ITG,*ne);

    nepar=0;
    for(i=0;i<*ne;i++){
	if(ipkon[i]>-1){

	    /* active element */
	    
	    ipar[nepar]=i;
	    nepar++;
	}
    }
    if(nepar<*num_cpus) *num_cpus=nepar;

    if(lakon==NULL){

	/* legacy: equal numbers of active elements */

	idelta=nepar/(*num_cpus);
	isum=0;
	for(i=0;i<*num_cpus;i++){
	    neapar[i]=ipar[isum];
	    if(i!=*num_cpus-1){
		isum+=idelta;
	    }else{
		isum=nepar;
	    }
	    nebpar[i]=ipar[isum-1];
	}

    }else{

	/* cost-weighted greedy split on prefix sums */

	NNEW(wpar,long long,nepar+1);
	wpar[0]=0;
	for(i=0;i<nepar;i++){
	    wpar[i+1]=wpar[i]+elem_cost(&lakon[8*ipar[i]]);
	}
	total=wpar[nepar];
	nthr=*num_cpus;
	isum=0;
	for(t=0;t<nthr;t++){
	    neapar[t]=ipar[isum];
	    if(t!=nthr-1){
		target=total*(t+1)/nthr;
		cum=isum;
		/* advance at least one element while elements remain
		   for the threads ahead */
		do{
		    cum++;
		}while(cum<nepar-(nthr-1-t)&&wpar[cum]<target);
		isum=cum;
	    }else{
		isum=nepar;
	    }
	    nebpar[t]=ipar[isum-1];
	}
	SFREE(wpar);
    }
    
    SFREE(ipar);
    
    return;
}
