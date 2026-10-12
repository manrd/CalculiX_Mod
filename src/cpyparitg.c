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

/*     Parallellization of the copying of an array 	 */

#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "CalculiX.h"

void cpyparitg(ITG *iva1,ITG *iva2,ITG *isize,ITG *num_cpus){
    ITG n = *isize;
    if(n<=0) return;
    if(*num_cpus<=1 || n<262144){
        memcpy(iva1, iva2, (size_t)n * sizeof(ITG));
        return;
    }
    #pragma omp parallel for schedule(static) num_threads(*num_cpus)
    for(ITG j=0;j<n;j++){
        iva1[j]=iva2[j];
    }
}

void *cpyparitgmt(ITG *i){
    return NULL;
}
