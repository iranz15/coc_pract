
static char help[] = "Computes the integral of 2*x/(1+x^2) in [0,1], which is equal to ln(2).\n\n";

/*
  Based on PETSc example $PETSC_DIR/src/vec/vec/tutorials/ex18.c

  Include "petscvec.h" so that we can use vectors.  Note that this file
  automatically includes:
     petscsys.h    - base PETSc routines
     petscis.h     - index sets
     petscviewer.h - viewers
*/
#include <petscvec.h>

/* This is the function for which we want to compute the integral */
PetscScalar func(PetscScalar a)
{
  return 2*a/(1+a*a);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscInt       rstart,rend,i,k,N=1000000;
  PetscScalar    result,value,h,*xarray;
  Vec            x,xend;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&N,NULL);CHKERRQ(ierr);
  
  
  
  /*
     Create parallel vector x of length N=number of subintervals
  */
  h = 1.0/N;
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecSet(x,0.0);CHKERRQ(ierr);

  /*
     Create vector xend by duplication, and fill its values.
     The xend vector is a dummy vector to find the value of the
     elements at the endpoints for use in the trapezoid rule.
  */
  ierr = VecDuplicate(x,&xend);CHKERRQ(ierr);

  value = 0.5;
  if (!rank) {
    i = 0;
    ierr = VecSetValues(xend,1,&i,&value,INSERT_VALUES);CHKERRQ(ierr);
  }
  if (rank == size-1) {
    i = N-1;
    ierr = VecSetValues(xend,1,&i,&value,INSERT_VALUES);CHKERRQ(ierr);
  }
  /*
     Assemble vector, using the 2-step process: VecAssemblyBegin(), VecAssemblyEnd()
  */
  ierr = VecAssemblyBegin(xend);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(xend);CHKERRQ(ierr);
  
  

  /*
     Set the x vector elements.
     i*h will return 0 for i=0 and 1 for i=N-1.
     The function evaluated (2x/(1+x^2)) is defined above.
     Each evaluation is put into the local array of the vector without message passing.
  */
  ierr = VecGetOwnershipRange(x,&rstart,&rend);CHKERRQ(ierr);
  ierr = VecGetArray(x,&xarray);CHKERRQ(ierr);
  k = 0;
  for (i=rstart; i<rend; i++) {
    xarray[k] = i*h;
    xarray[k] = func(xarray[k]);
    k++;
  }
  ierr = VecRestoreArray(x,&xarray);CHKERRQ(ierr);

  /*
     Evaluates the integral.  First the sum of all the points is taken.
     That result is multiplied by the step size for the trapezoid rule.
     Then half the value at each endpoint is subtracted,
     this is part of the composite trapezoid rule.
  */
  ierr   = VecSum(x,&result);CHKERRQ(ierr);
  result = result*h;
  ierr   = VecDot(x,xend,&value);CHKERRQ(ierr);
  result = result-h*value;

  /*
      Print the value of the integral and destroy objects
  */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"ln(2) is %g\n",(double)result);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&xend);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

