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

/*     parallellization of the calculation of the forces in resultsforc.c */

#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include "CalculiX.h"

void forparll(ITG *mt,ITG *nactdof,double *f,double *fn,
		    ITG *nk,ITG *num_cpus){
    ITG n = *nk;
    ITG m = *mt;
    if(n<=0) return;
    if(*num_cpus<=1 || n<16384){
        for(ITG k=0;k<n;++k){
            for(ITG j=0;j<m;++j){
                ITG idof = nactdof[m*k+j];
                if(idof>0) f[idof-1]=fn[m*k+j];
            }
        }
        return;
    }
    #pragma omp parallel for schedule(static) num_threads(*num_cpus)
    for(ITG k=0;k<n;++k){
        for(ITG j=0;j<m;++j){
            ITG idof = nactdof[m*k+j];
            if(idof>0) f[idof-1]=fn[m*k+j];
        }
    }
}

void *forparllmt(ITG *i){
    return NULL;
}
