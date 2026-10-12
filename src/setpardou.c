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

/*     Parallellization of assigning double values to a field	 */

#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "CalculiX.h"

void setpardou(double *var1,double var2,ITG isize,ITG num_cpus){
    if(isize<=0) return;
    if(var2==0.0 && (num_cpus<=1 || isize<262144)){
        memset(var1, 0, (size_t)isize * sizeof(double));
        return;
    }
    if(num_cpus<=1 || isize<65536){
        for(ITG j=0;j<isize;j++) var1[j]=var2;
        return;
    }
    #pragma omp parallel for schedule(static) num_threads(num_cpus)
    for(ITG j=0;j<isize;j++){
        var1[j]=var2;
    }
}

void *setpardoumt(ITG *i){
    return NULL;
}
