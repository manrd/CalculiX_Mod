/*     CalculiX - A 3-dimensional finite element program                   */
/*              Copyright (C) 1998-2005 Guido Dhondt                          */

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

#ifdef PARDISO

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "CalculiX.h"
#include "pardiso.h"
#include "ccx_prof.h"

/* next line is for the simultaneous use of PARDISO and PaStiX */

#include <mkl_service.h>

ITG *icolpardiso=NULL,*pointers=NULL,iparm[64];
long long pt[64];
double *aupardiso=NULL;
float *aupardiso_f32=NULL;
/* double dparm[64];  not used */
ITG mthread_mkl=0;
char envMKL[32];

/* Symbolic factorization cache state across nonlinear/contact iterations */
static ITG pardiso_has_symb = 0;
static ITG prev_neq = 0, prev_nzs = 0, prev_sym = -1, prev_fmt = -1, prev_mtype = 0, prev_ndim = 0;
static ITG *prev_icol = NULL, *prev_irow = NULL;
static ITG *lower_perm_map = NULL;
static int use_fp32_mode = -1;

static int pardiso_check_fp32(void) {
  if (use_fp32_mode < 0) {
    const char *e = getenv("CCX_PARDISO_FP32");
    use_fp32_mode = (e && e[0] != '\0' && e[0] != '0') ? 1 : 0;
  }
  return use_fp32_mode;
}

static void pardiso_release_cached(void) {
  if (!pardiso_has_symb) return;
  ITG maxfct=1, mnum=1, phase=-1, nrhs=1, msglvl=0, error=0;
  ITG *perm=NULL;
  double *b=NULL, *x=NULL;
  void *val_ptr = pardiso_check_fp32() ? (void*)aupardiso_f32 : (void*)aupardiso;
  FORTRAN(pardiso,(pt,&maxfct,&mnum,&prev_mtype,&phase,&prev_neq,val_ptr,
                   pointers,icolpardiso,perm,&nrhs,iparm,&msglvl,
                   b,x,&error));
  SFREE(icolpardiso);
  SFREE(aupardiso);
  SFREE(aupardiso_f32);
  SFREE(pointers);
  SFREE(prev_icol);
  SFREE(prev_irow);
  SFREE(lower_perm_map);
  pardiso_has_symb = 0;
  prev_neq = 0;
  prev_nzs = 0;
  prev_sym = -1;
  prev_fmt = -1;
  prev_mtype = 0;
  prev_ndim = 0;
}

void pardiso_factor(double *ad, double *au, double *adb, double *aub, 
		    double *sigma,ITG *icol, ITG *irow, 
		    ITG *neq, ITG *nzs, ITG *symmetryflag, ITG *inputformat,
		    ITG *jq, ITG *nzs3){

  char *env;
  ITG i,j,k,l,maxfct=1,mnum=1,nrhs=1,*perm=NULL,mtype,
    msglvl=0,error=0,k2;
  ITG ndim,nthread,nthread_v;
  double *b=NULL,*x=NULL;
  int fp32 = pardiso_check_fp32();

  if(*symmetryflag==0){
    printf(" Factoring the system of equations using the symmetric pardiso solver\n");
    mtype=-2;
  }else{
    printf(" Factoring the system of equations using the unsymmetric pardiso solver\n");
    if(*inputformat==3){
      mtype=11;
    }else{
      mtype=1;
    }
  }

  /* set MKL_NUM_THREADS to min(CCX_NPROC_EQUATION_SOLVER,OMP_NUM_THREADS)
     must be done once  */
  if (mthread_mkl == 0) {
    nthread=1;
    env=getenv("MKL_NUM_THREADS");
    if(env) {
      nthread=atoi(env);}
    else {
      env=getenv("OMP_NUM_THREADS");
      if(env) {nthread=atoi(env);}
    }
    env=getenv("CCX_NPROC_EQUATION_SOLVER");
    if(env) {
      nthread_v=atoi(env);
      if (nthread_v <= nthread) {nthread=nthread_v;}
    }
    if (nthread < 1) {nthread=1;}
    sprintf(envMKL,"MKL_NUM_THREADS=%" ITGFORMAT "",nthread);  
    putenv(envMKL);
    mthread_mkl=nthread;
  }
    
  printf(" number of threads =% d\n\n",mthread_mkl);

  double _t_conv0 = ccx_prof_ms();
  double _t_sort0 = _t_conv0, _t_sort1 = _t_conv0;

  /* Check if we can reuse the symbolic factorization (phase 11) from the previous iteration */
  int reuse_symb = 0;
  const char *env_nocache = getenv("CCX_PARDISO_NOCACHE");
  int allow_cache = !(env_nocache && env_nocache[0] != '\0' && env_nocache[0] != '0');

  if (allow_cache && pardiso_has_symb &&
      *neq == prev_neq && *nzs == prev_nzs &&
      *symmetryflag == prev_sym && *inputformat == prev_fmt &&
      mtype == prev_mtype && *inputformat != 3 &&
      prev_icol != NULL && prev_irow != NULL) {
    if (memcmp(icol, prev_icol, (size_t)(*neq) * sizeof(ITG)) == 0 &&
        memcmp(irow, prev_irow, (size_t)(*nzs) * sizeof(ITG)) == 0) {
      reuse_symb = 1;
    }
  }

  if (!reuse_symb) {
    if (pardiso_has_symb) {
      pardiso_release_cached();
    }
    for(i=0;i<64;i++){pt[i]=0; iparm[i]=0;}
    /* Properly configure MKL PARDISO iparm with iparm[0]=1 so iparm[1]=3 is respected! */
    iparm[0] = 1; /* Use custom iparm values */
    iparm[1] = (mthread_mkl > 1) ? 3 : 2; /* 3 = OpenMP parallel METIS nested dissection */
    iparm[9] = (mtype == 11) ? 13 : 8;    /* Pivoting perturbation 10^-8 (sym) / 10^-13 (unsym) */
    iparm[10] = (mtype == 11) ? 1 : 0;    /* Scaling on for unsymm, off for symm */
    iparm[12] = (mtype == 11) ? 1 : 0;    /* Weighted matching on for unsymm, off for symm */
    iparm[17] = -1;                       /* Report nnz in factors */
    iparm[20] = (mtype == -2) ? 1 : 0;    /* 1x1 and 2x2 Bunch-Kaufman pivoting */
    iparm[23] = (mtype == 11 && mthread_mkl >= 8) ? 1 : 0; /* Two-level parallel factorization for unsym >=8 threads */
    iparm[24] = (mthread_mkl > 1) ? 2 : 0; /* Parallel forward/backward solve */
    iparm[27] = fp32 ? 1 : 0;             /* 0 = double precision, 1 = single precision (float32) */
  }

  if(*symmetryflag==0){

    /* symmetric matrix; the subdiagonal entries are stored
       column by column in au, the diagonal entries in ad;
       pardiso needs the entries row per row */      

    ndim=*neq+*nzs;
      
    if (!reuse_symb) {
      NNEW(pointers,ITG,*neq+1);
      NNEW(icolpardiso,ITG,ndim);
      NNEW(aupardiso,double,ndim);
      if (fp32) NNEW(aupardiso_f32,float,ndim);

      pointers[0] = 1;
      for(i=0;i<*neq;i++){
        pointers[i+1] = pointers[i] + icol[i] + 1;
      }

      if(*sigma==0.){
        #pragma omp parallel for schedule(static) num_threads(mthread_mkl)
        for(ITG row=0; row<*neq; row++){
          ITG k_idx = pointers[row] - 1;
          ITG l_idx = k_idx - row;
          ITG len = icol[row];
          icolpardiso[k_idx] = row + 1;
          aupardiso[k_idx] = ad[row];
          k_idx++;
          for(ITG jj=0; jj<len; jj++){
            icolpardiso[k_idx + jj] = irow[l_idx + jj];
            aupardiso[k_idx + jj] = au[l_idx + jj];
          }
        }
      }else{
        double sig = *sigma;
        #pragma omp parallel for schedule(static) num_threads(mthread_mkl)
        for(ITG row=0; row<*neq; row++){
          ITG k_idx = pointers[row] - 1;
          ITG l_idx = k_idx - row;
          ITG len = icol[row];
          icolpardiso[k_idx] = row + 1;
          aupardiso[k_idx] = ad[row] - sig * adb[row];
          k_idx++;
          for(ITG jj=0; jj<len; jj++){
            icolpardiso[k_idx + jj] = irow[l_idx + jj];
            aupardiso[k_idx + jj] = au[l_idx + jj] - sig * aub[l_idx + jj];
          }
        }
      }
    } else {
      /* Fast value-only update when reusing symbolic structure */
      if(*sigma==0.){
        #pragma omp parallel for schedule(static) num_threads(mthread_mkl)
        for(ITG row=0; row<*neq; row++){
          ITG k_idx = pointers[row] - 1;
          ITG l_idx = k_idx - row;
          ITG len = icol[row];
          aupardiso[k_idx] = ad[row];
          k_idx++;
          for(ITG jj=0; jj<len; jj++){
            aupardiso[k_idx + jj] = au[l_idx + jj];
          }
        }
      }else{
        double sig = *sigma;
        #pragma omp parallel for schedule(static) num_threads(mthread_mkl)
        for(ITG row=0; row<*neq; row++){
          ITG k_idx = pointers[row] - 1;
          ITG l_idx = k_idx - row;
          ITG len = icol[row];
          aupardiso[k_idx] = ad[row] - sig * adb[row];
          k_idx++;
          for(ITG jj=0; jj<len; jj++){
            aupardiso[k_idx + jj] = au[l_idx + jj] - sig * aub[l_idx + jj];
          }
        }
      }
    }
  }else{

    if(*inputformat==3){

      /* off-diagonal terms are stored column per column from top to bottom in au;
	 diagonal terms are stored in ad.
	 O(nzs) 2-pass counting-sort transpose (zero isortiid/isortid needed!) */

      mtype=11;
      NNEW(pointers,ITG,*neq+1);
      ITG *row_counts = NULL;
      NNEW(row_counts,ITG,*neq);

      /* Pass 1: count nonzeros per row (including diagonal) */
      for(i=0;i<*neq;i++) row_counts[i] = 1;
      k=0;
      for(i=0;i<*neq;i++){
	for(j=0;j<icol[i];j++){
	  if(au[k]>1.e-12||au[k]<-1.e-12){
	    row_counts[irow[k]-1]++;
	  }
	  k++;
	}
      }
      pointers[0]=1;
      for(i=0;i<*neq;i++){
	pointers[i+1]=pointers[i]+row_counts[i];
	row_counts[i]=pointers[i]-1; /* reuse row_counts as write_pos */
      }
      ndim=pointers[*neq]-1;
      NNEW(icolpardiso,ITG,ndim);
      NNEW(aupardiso,double,ndim);
      if (fp32) NNEW(aupardiso_f32,float,ndim);

      /* Pass 2: iterate columns i=0..*neq-1 in strictly ascending order.
         Since columns are visited in ascending order, every row's column indices
         are automatically inserted in strictly sorted order! */
      k=0;
      for(i=0;i<*neq;i++){
	ITG len=icol[i];
	ITG diag_inserted=0;
	for(j=0;j<len;j++){
	  double val=au[k];
	  ITG r=irow[k]-1;
	  k++;
	  if(val>1.e-12||val<-1.e-12){
	    if(!diag_inserted && r>i){
	      ITG dpos=row_counts[i]++;
	      icolpardiso[dpos]=i+1;
	      aupardiso[dpos]=ad[i];
	      diag_inserted=1;
	    }
	    ITG pos=row_counts[r]++;
	    icolpardiso[pos]=i+1;
	    aupardiso[pos]=val;
	  }
	}
	if(!diag_inserted){
	  ITG dpos=row_counts[i]++;
	  icolpardiso[dpos]=i+1;
	  aupardiso[dpos]=ad[i];
	}
      }
      SFREE(row_counts);

    }else if(*inputformat==1){
	  
      /* lower triangular matrix is stored column by column in
	 au, followed by the upper triangular matrix row by row;
	 the diagonal terms are stored in ad.
	 O(nzs) counting-sort transpose replaces isortiid + isortid! */

      mtype=1;
      ndim=*neq+2**nzs;

      if (!reuse_symb) {
        NNEW(pointers,ITG,*neq+1);
        NNEW(icolpardiso,ITG,ndim);
        NNEW(aupardiso,double,ndim);
        if (fp32) NNEW(aupardiso_f32,float,ndim);
        NNEW(lower_perm_map,ITG,*nzs);

        ITG *row_counts = NULL;
        NNEW(row_counts,ITG,*neq);
        for(k=0;k<*nzs;k++){
          row_counts[irow[k]-1]++;
        }
        pointers[0]=1;
        for(i=0;i<*neq;i++){
          pointers[i+1]=pointers[i]+row_counts[i]+1+icol[i];
          row_counts[i]=pointers[i]-1; /* write_pos for lower triangle of row i */
        }

        /* Scatter lower-triangle entries in ascending column order -> automatically sorted! */
        k=0;
        for(i=0;i<*neq;i++){
          ITG len=icol[i];
          for(j=0;j<len;j++){
            ITG r=irow[k]-1;
            ITG pos=row_counts[r]++;
            icolpardiso[pos]=i+1;
            aupardiso[pos]=au[k];
            lower_perm_map[k]=pos;
            k++;
          }
        }

        /* Fill diagonal and upper triangle for each row in parallel */
        #pragma omp parallel for schedule(static) num_threads(mthread_mkl)
        for(ITG r=0; r<*neq; r++){
          ITG pos = row_counts[r];
          icolpardiso[pos] = r + 1;
          aupardiso[pos] = ad[r];
          pos++;
          ITG jstart = jq[r] - 1;
          ITG jend = jq[r+1] - 1;
          for(ITG jj=jstart; jj<jend; jj++){
            icolpardiso[pos] = irow[jj];
            aupardiso[pos] = au[jj + *nzs3];
            pos++;
          }
        }
        SFREE(row_counts);
      } else {
        /* Reusing symbolic structure for inputformat==1: update values in parallel */
        #pragma omp parallel for schedule(static) num_threads(mthread_mkl)
        for(ITG k_idx=0; k_idx<*nzs; k_idx++){
          aupardiso[lower_perm_map[k_idx]] = au[k_idx];
        }
        #pragma omp parallel for schedule(static) num_threads(mthread_mkl)
        for(ITG r=0; r<*neq; r++){
          ITG jstart = jq[r] - 1;
          ITG jend = jq[r+1] - 1;
          ITG len = jend - jstart;
          ITG pos = (pointers[r+1] - 1) - len - 1;
          aupardiso[pos] = ad[r];
          pos++;
          for(ITG jj=jstart; jj<jend; jj++){
            aupardiso[pos++] = au[jj + *nzs3];
          }
        }
      }
    }
  }

  if (fp32) {
    #pragma omp parallel for schedule(static) num_threads(mthread_mkl)
    for(ITG idx=0; idx<ndim; idx++){
      aupardiso_f32[idx] = (float)aupardiso[idx];
    }
  }
  double _t_conv1 = ccx_prof_ms();

  mkl_domain_set_num_threads(mthread_mkl,MKL_DOMAIN_PARDISO);
  void *val_ptr = fp32 ? (void*)aupardiso_f32 : (void*)aupardiso;

  double _t_p11 = _t_conv1;
  if (!reuse_symb) {
    ITG phase11 = 11;
    FORTRAN(pardiso,(pt,&maxfct,&mnum,&mtype,&phase11,neq,val_ptr,
		     pointers,icolpardiso,perm,&nrhs,iparm,&msglvl,
                     b,x,&error));
    _t_p11 = ccx_prof_ms();
    if (allow_cache && *inputformat != 3) {
      NNEW(prev_icol,ITG,*neq);
      NNEW(prev_irow,ITG,*nzs);
      memcpy(prev_icol, icol, (size_t)(*neq) * sizeof(ITG));
      memcpy(prev_irow, irow, (size_t)(*nzs) * sizeof(ITG));
      prev_neq = *neq;
      prev_nzs = *nzs;
      prev_sym = *symmetryflag;
      prev_fmt = *inputformat;
      prev_mtype = mtype;
      prev_ndim = ndim;
      pardiso_has_symb = 1;
    }
  }

  ITG phase22 = 22;
  FORTRAN(pardiso,(pt,&maxfct,&mnum,&mtype,&phase22,neq,val_ptr,
		   pointers,icolpardiso,perm,&nrhs,iparm,&msglvl,
                   b,x,&error));
  double _t_fac1 = ccx_prof_ms();
  CCX_PROF_LOG("pardiso_factor: csr_conv=%.2f ms, sort=%.2f ms, mkl_phase11=%.2f ms, mkl_phase22=%.2f ms [sym=%d, fmt=%d, neq=%d, ndim=%d, nt=%d, reuse=%d, fp32=%d]",
               _t_conv1 - _t_conv0, _t_sort1 - _t_sort0, _t_p11 - _t_conv1, _t_fac1 - _t_p11, (int)*symmetryflag, (int)*inputformat, (int)*neq, (int)ndim, (int)mthread_mkl, reuse_symb, fp32);

  return;
}

/* Helper to compute r = b - A * x in double precision for mixed-precision iterative refinement */
static void pardiso_csr_residual_f64(ITG neq, ITG sym, const ITG *ia, const ITG *ja, const double *a,
                                     const double *b, const double *x, double *r, ITG nthreads) {
  if (sym == 0) {
    /* Upper triangular CSR: diagonal is at k = ia[i]-1, upper off-diagonals at k = ia[i] .. ia[i+1]-2. */
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (ITG i = 0; i < neq; i++) {
      r[i] = b[i];
    }
    #pragma omp parallel for schedule(static, 256) num_threads(nthreads)
    for (ITG i = 0; i < neq; i++) {
      ITG kstart = ia[i] - 1;
      ITG kend = ia[i+1] - 1;
      double xi = x[i];
      double sum = a[kstart] * xi;
      for (ITG k = kstart + 1; k < kend; k++) {
        ITG col = ja[k] - 1;
        double val = a[k];
        sum += val * x[col];
        double contrib = val * xi;
        #pragma omp atomic
        r[col] -= contrib;
      }
      #pragma omp atomic
      r[i] -= sum;
    }
  } else {
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (ITG i = 0; i < neq; i++) {
      ITG kstart = ia[i] - 1;
      ITG kend = ia[i+1] - 1;
      double sum = 0.0;
      for (ITG k = kstart; k < kend; k++) {
        sum += a[k] * x[ja[k] - 1];
      }
      r[i] = b[i] - sum;
    }
  }
}

void pardiso_solve(double *b, ITG *neq,ITG *symmetryflag,ITG *inputformat,
		   ITG *nrhs){

  ITG maxfct=1,mnum=1,phase=33,*perm=NULL,mtype,
    msglvl=0,i,error=0;
  double *x=NULL;
  double _t_s0 = ccx_prof_ms();
  int fp32 = pardiso_check_fp32();

  if(*symmetryflag==0){
    mtype=-2;
  }else{
    if(*inputformat==3){
      mtype=11;
    }else{
      mtype=1;
    }
  }

  if (!fp32) {
    NNEW(x,double,*nrhs**neq);

    FORTRAN(pardiso,(pt,&maxfct,&mnum,&mtype,&phase,neq,aupardiso,
		     pointers,icolpardiso,perm,nrhs,iparm,&msglvl,
                     b,x,&error));

    #pragma omp parallel for schedule(static) num_threads(mthread_mkl)
    for(i=0;i<*nrhs**neq;i++){b[i]=x[i];}
    SFREE(x);
  } else {
    /* Mixed-precision solve: float32 PARDISO phase 33 + float64 iterative refinement */
    ITG len = *nrhs * *neq;
    float *b_f32 = NULL, *x_f32 = NULL;
    double *x_d64 = NULL, *r_d64 = NULL;
    NNEW(b_f32, float, len);
    NNEW(x_f32, float, len);
    NNEW(x_d64, double, len);
    NNEW(r_d64, double, len);

    double b_norm2 = 0.0;
    #pragma omp parallel for reduction(+:b_norm2) schedule(static) num_threads(mthread_mkl)
    for(i=0;i<len;i++){
      b_f32[i] = (float)b[i];
      b_norm2 += b[i] * b[i];
    }
    double b_norm = sqrt(b_norm2);

    FORTRAN(pardiso,(pt,&maxfct,&mnum,&mtype,&phase,neq,aupardiso_f32,
		     pointers,icolpardiso,perm,nrhs,iparm,&msglvl,
                     b_f32,x_f32,&error));

    #pragma omp parallel for schedule(static) num_threads(mthread_mkl)
    for(i=0;i<len;i++){
      x_d64[i] = (double)x_f32[i];
    }

    if (*nrhs == 1 && b_norm > 0.0) {
      for (int ir = 0; ir < 3; ir++) {
        pardiso_csr_residual_f64(*neq, *symmetryflag, pointers, icolpardiso, aupardiso, b, x_d64, r_d64, mthread_mkl);
        double r_norm2 = 0.0;
        #pragma omp parallel for reduction(+:r_norm2) schedule(static) num_threads(mthread_mkl)
        for(i=0;i<*neq;i++){
          r_norm2 += r_d64[i] * r_d64[i];
        }
        double rel_res = sqrt(r_norm2) / b_norm;
        if (rel_res < 1e-10) break;
        double scale = sqrt(r_norm2);
        double inv_scale = 1.0 / scale;
        #pragma omp parallel for schedule(static) num_threads(mthread_mkl)
        for(i=0;i<*neq;i++){
          b_f32[i] = (float)(r_d64[i] * inv_scale);
        }
        FORTRAN(pardiso,(pt,&maxfct,&mnum,&mtype,&phase,neq,aupardiso_f32,
                         pointers,icolpardiso,perm,nrhs,iparm,&msglvl,
                         b_f32,x_f32,&error));
        #pragma omp parallel for schedule(static) num_threads(mthread_mkl)
        for(i=0;i<*neq;i++){
          x_d64[i] += scale * (double)x_f32[i];
        }
      }
    }

    #pragma omp parallel for schedule(static) num_threads(mthread_mkl)
    for(i=0;i<len;i++){b[i]=x_d64[i];}

    SFREE(b_f32);
    SFREE(x_f32);
    SFREE(x_d64);
    SFREE(r_d64);
  }
  double _t_s1 = ccx_prof_ms();
  CCX_PROF_LOG("pardiso_solve: mkl_phase33=%.2f ms", _t_s1 - _t_s0);

  return;
}

void pardiso_cleanup(ITG *neq,ITG *symmetryflag,ITG *inputformat){
  const char *env_nocache = getenv("CCX_PARDISO_NOCACHE");
  if (env_nocache && env_nocache[0] != '\0' && env_nocache[0] != '0') {
    pardiso_release_cached();
    return;
  }
  /* If symbolic caching is active (inputformat != 3), keep pt, pointers, icolpardiso, aupardiso
     alive for potential reuse in the next nonlinear/contact iteration! */
  if (pardiso_has_symb && *inputformat != 3) {
    return;
  }

  ITG maxfct=1,mnum=1,phase=-1,*perm=NULL,nrhs=1,mtype,
    msglvl=0,error=0;
  double *b=NULL,*x=NULL;

  if(*symmetryflag==0){
    mtype=-2;
  }else{
    if(*inputformat==3){
      mtype=11;
    }else{
      mtype=1;
    }
  }

  void *val_ptr = pardiso_check_fp32() ? (void*)aupardiso_f32 : (void*)aupardiso;
  FORTRAN(pardiso,(pt,&maxfct,&mnum,&mtype,&phase,neq,val_ptr,
		   pointers,icolpardiso,perm,&nrhs,iparm,&msglvl,
                   b,x,&error));

  SFREE(icolpardiso);
  SFREE(aupardiso);
  SFREE(aupardiso_f32);
  SFREE(pointers);

  return;
}

void pardiso_main(double *ad, double *au, double *adb, double *aub, 
		  double *sigma,double *b, ITG *icol, ITG *irow, 
		  ITG *neq, ITG *nzs,ITG *symmetryflag,ITG *inputformat,
		  ITG *jq, ITG *nzs3,ITG *nrhs){

  if(*neq==0) return;

  pardiso_factor(ad,au,adb,aub,sigma,icol,irow, 
		 neq,nzs,symmetryflag,inputformat,jq,nzs3);

  pardiso_solve(b,neq,symmetryflag,inputformat,nrhs);

  pardiso_cleanup(neq,symmetryflag,inputformat);

  return;
}

#endif
